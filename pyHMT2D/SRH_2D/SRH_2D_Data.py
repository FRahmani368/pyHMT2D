# -*- coding: utf-8 -*-
"""

Author: Xiaofeng Liu, PhD, PE
Penn State University
"""

import sys
import os
import copy
import numpy as np
import h5py
import meshio
from scipy import interpolate
from osgeo import gdal
from os import path
import shlex
import vtk

from .helpers import *
from ..__common__ import *

# map from number of nodes to cell type in VTK (see VTK documentation)
vtkCellTypeMap = {
  3: 5,     # triangle
  4: 9,     # quad
  5: 7,     # poly
  6: 7,
  7: 7,
  8: 7
}

# maximum number of nodes for an element
gMax_Nodes_per_Element = 8

# maximum number of elements for a node
gMax_Elements_per_Node = 10

class SRH_2D_SRHHydro:
    """A class to handle srhhydro file for SRH-2D

    Attributes:


    Methods:

    """
    def __init__(self, srhhydro_filename):
        self.srhhydro_filename = srhhydro_filename #srhhydro file name

        #dict to hold the content of the srhhydro file
        self.srhhydro_content = {}

        #parse the srhhydro file and build srhhydro_content
        self.parse_srhhydro_file()

    def parse_srhhydro_file(self):
        res_all = {}
        res_ManningsN = {}  # dict for ManningsN (there cuould be multiple entries)
        res_BC = {}  # dict for BC (there could be multiple entries)
        res_IQParams = {}  # dict for subcritical inlet discharge boundary condition
        res_ISupCrParams = {}  # dict for supercritical inlet, the same as IQParams, with the additon of WSE
        res_EWSParamsC = {}  # dict for stage exit boundary condition (constant)
        res_EWSParamsRC = {}  # dict for stage exit boundary condition (rating curve)
        res_EQParams = {}  # dict for exit discharge boundary condition
        res_NDParams = {}  # dict for normal depth outlet boundary condition

        #check whether the srhhydro file exists
        if not path.isfile(self.srhhydro_filename):
            print("The SRHHYDRO file", self.srhhydro_filename, "does not exists. Exitting ...")
            sys.exit()

        for line in open(self.srhhydro_filename):
            #parts = line.strip().split(' ')
            parts = shlex.split(line.strip())

            #print(parts)

            map(str.strip, parts)

            #print(parts)

            if len(parts) <= 1:  # if there is only one word, assume it is a comment; do nothing
                continue

            if parts[0] == 'ManningsN':
                res_ManningsN[int(parts[1])] = float(parts[2])
            elif parts[0] == 'BC':
                res_BC[int(parts[1])] = parts[2]
            elif parts[0] == 'IQParams':
                res_IQParams[int(parts[1])] = [parts[2], parts[3], parts[4]]
            elif parts[0] == 'ISupCrParams': #need to check these
                res_ISupCrParams[int(parts[1])] = parts[2]
            elif parts[0] == 'EWSParamsC':
                res_EWSParamsC[int(parts[1])] = [parts[2], parts[3], parts[4]]
            elif parts[0] == 'EWSParamsRC':
                res_EWSParamsRC[int(parts[1])] = [parts[2], parts[3], parts[4]]
            elif parts[0] == 'EQParams': #need to check these
                res_EQParams[int(parts[1])] = parts[2]
            elif parts[0] == 'NDParams': #need to check these
                res_NDParams[int(parts[1])] = parts[2]
            elif parts[0] == 'OutputFormat':
                res_all[parts[0]] = [parts[1],parts[2]]
            elif parts[0] == 'SimTime':
                res_all[parts[0]] = [float(parts[1]),float(parts[2]),float(parts[3])]
            elif parts[0] == 'ParabolicTurbulence':
                res_all[parts[0]] = float(parts[1])
            elif parts[0] == 'OutputOption':
                res_all[parts[0]] = int(parts[1])
            elif parts[0] == 'OutputInterval':
                res_all[parts[0]] = float(parts[1])
            else:
                #res_all[parts[0].lstrip()] = parts[1].strip('"').split(' ')
                res_all[parts[0].lstrip()] = parts[1]

        # add ManningsN, BC and all other sub-dict to res_all
        if res_ManningsN:
            res_all['ManningsN'] = res_ManningsN

        if res_BC:
            res_all['BC'] = res_BC

        if res_IQParams:
            res_all['IQParams'] = res_IQParams

        if res_ISupCrParams:
            res_all['ISupCrParams'] = res_ISupCrParams

        if res_EWSParamsC:
            res_all['EWSParamsC'] = res_EWSParamsC

        if res_EWSParamsRC:
            res_all['EWSParamsRC'] = res_EWSParamsRC

        if res_EQParams:
            res_all['EQParams'] = res_EQParams

        if res_NDParams:
            res_all['NDParams'] = res_NDParams


        if False:
            print(res_all)

            if res_ManningsN: print(res_all['ManningsN'])
            if res_BC: print(res_all['BC'])
            if res_IQParams: print(res_all['IQParams'])
            if res_ISupCrParams: print(res_all['ISupCrParams'])
            if res_EWSParamsC: print(res_all['EWSParamsC'])
            if res_EWSParamsRC: print(res_all['EWSParamsRC'])
            if res_EQParams: print(res_all['EQParams'])
            if res_NDParams: print(res_all['NDParams'])

        self.srhhydro_content = res_all

    def modify_ManningsN(self, materialID, newManningsNValue):
        """Modify materialID's Manning's n value to newManningsNValue

        Parameters
        ----------
        materialID: {int} -- material ID
        newManningsNValue: {float} -- new Manning's n value

        Returns
        -------

        """
        print("Modify Manning's n value ...")

        if not isinstance(materialID, int):
            print("Material ID has to be an integer. The type of materialID passed in is ", type(materialID),
                  ". Exit.\n")

        if not isinstance(newManningsNValue, float):
            print("Manning's n has to be a float. The type of newManningsNValue passed in is ", type(newManningsNValue),
                  ". Exit.\n")


        nDict = self.srhhydro_content["ManningsN"]

        if materialID in nDict:
            print("    Old Manning's n value =", nDict[materialID], "for material ID = ", materialID)
            nDict[materialID] = newManningsNValue
            print("    New Manning's n value =", nDict[materialID], "for material ID = ", materialID)
        else:
            print("The specified materialID", materialID, "is not in the Manning's n list. Please check.")



    def write_to_file(self, new_srhhydro_file_name):
        """Write to a SRHHYDRO file (useful for modification of the SRHHYDRO file)

        Returns
        -------

        """

        print("Wring the SRHHYDRO file %s \n" % new_srhhydro_file_name)

        try:
            fid = open(new_srhhydro_file_name, 'w')
        except IOError:
            print('srhhydro file open error')
            sys.exit()

        for key, value in self.srhhydro_content.items():
            #print("key, value = ", key, value)
            if type(value) is dict: #if the current item is dictionary itself
                for subkey, subvalue in value.items():
                    if "ManningsN" in key:
                        fid.write("ManningsN " + str(subkey) + ' ' + str(subvalue) + '\n')
                    elif "BC" in key:
                        fid.write("BC " + str(subkey) + ' ' + str(subvalue) + '\n')
                    elif "IQParams" in key:
                        fid.write("IQParams " + str(subkey) + ' ' + str(subvalue[0]) + ' ' + str(subvalue[1])+ ' ' +
                                  str(subvalue[2]) + '\n' )
                    elif "EWSParamsC" in key:
                        fid.write("EWSParamsC " + str(subkey) + ' ' + str(subvalue[0])+ ' ' + str(subvalue[1])+ ' ' +
                                  str(subvalue[2]) + '\n' )
                    elif "EWSParamsRC" in key:
                        fid.write("EWSParamsRC " + str(subkey) + ' \"' + str(subvalue[0])+ '\" ' + str(subvalue[1])+
                                  ' ' + str(subvalue[2]) + '\n' )
                    elif "ISupCrParams" in key:  #need to check these (no reference)
                        fid.write("ISupCrParams " + str(subkey) + ' ' + str(subvalue) + '\n' )
                    elif "EQParams" in key:      #need to check these (no reference)
                        fid.write("EQParams " + str(subkey) + ' ' + str(subvalue) + '\n' )
                    elif "NDParams" in key:      #need to check these (no reference)
                        fid.write("NDParams " + str(subkey) + ' ' + str(subvalue) + '\n' )
            else:
                if "Case" in key or "Description" in key or "Grid" in key or "HydroMat" in key:
                    fid.write(str(key) + ' \"' + str(value) + '\"\n')
                elif "OutputFormat" in key:
                    fid.write(str(key) + ' ' + str(value[0]) + ' ' + str(value[1]) + '\n')
                elif "SimTime" in key:
                    fid.write(str(key) + ' ' + str(value[0]) + ' ' + str(value[1]) + ' ' + str(value[2]) + '\n')
                else:
                    #print("last, key, value", key, value, str(key), str(value))
                    fid.write(str(key) + ' ' + str(value) + '\n')

        fid.close()

    def get_simulation_start_end_time(self):
        """Get the start and end time of simulation (in hours, from the "SimTime" entry)

        Returns
        -------
        startTime (hr)
        endTime (hr)

        """

        value = self.srhhydro_content["SimTime"]

        return value[0],value[2]

    def get_simulation_time_step_size(self):
        """Get the time step size of simulation (in seconds, from the "SimTime" entry)

        Returns
        -------
        deltaT (s)

        """

        value = self.srhhydro_content["SimTime"]

        return value[1]

class SRH_2D_SRHGeom:
    """A class to handle srhgeom file for SRH-2D

    Attributes:


    Methods:

    """
    def __init__(self, srhgeom_filename):
        self.srhgeom_filename = srhgeom_filename

        #mesh information:
        #   number of elements
        self.numOfElements = -1

        #   number of nodes
        self.numOfNodes = -1

        # number of NodeString
        self.numOfNodeStrings = -1

        # First read of the SRHGEOM file to get number of elements and nodes
        self.getNumOfElementsNodes()

        # list of nodes for all elements
        self.elementNodesList = np.zeros([self.numOfElements, gMax_Nodes_per_Element], dtype=int)

        # number of nodes for each element (3,4,...,gMax_Nodes_per_Element)
        self.elementNodesCount = np.zeros(self.numOfElements, dtype=int)

        # each element's vtk cell type
        self.vtkCellTypeCode = np.zeros(self.numOfElements, dtype=int)

        # each node's 3D coordinates
        self.nodeCoordinates = np.zeros([self.numOfNodes, 3], dtype=np.float64)

        # each element (cell)'s bed elevation (z)
        self.elementBedElevation = np.zeros(self.numOfElements, dtype=np.float64)

        # each NodeString's list of nodes (stored in a dictionary)
        self.nodeStringsDict = {}

        # get the mesh information (elementNodesList, elementNodesCount, vtkCellTypeCode, and nodeCoordinates)
        # from reading the SRHGEOM file again
        self.readSRHGEOMFile()

        # list of elements for all nodes
        self.nodeElementsList = np.zeros([self.numOfNodes, gMax_Elements_per_Node], dtype=int)

        # number of elements for each node (1,2,...,gMax_Elements_per_Node)
        self.nodeElementsCount = np.zeros(self.numOfNodes, dtype=int)

        # build the element list for each node
        self.buildNodeElements()


    def getNumOfElementsNodes(self):
        """ Get the number of elements and nodes in srhgeom mesh file

        Returns
        -------

        """
        print("Getting numbers of elements and nodes from the SRHGEOM file ...")

        # read the "srhgeom" mesh file
        try:
            srhgeomfile = open(self.srhgeom_filename, 'r')
        except:
            print('Failed openning the SRHGEOM file', self.srhgeom_filename)
            sys.exit()

        count = 0
        elemCount = 0
        nodeCount = 0
        nodeStringCount = 0

        while True:
            count += 1

            # Get next line from file
            line = srhgeomfile.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break

            # print("Line{}: {}".format(count, line.strip()))

            search = line.split()
            # print(search)

            if len(search) != 0:
                if search[0] == "Elem":
                    elemCount += 1
                    # print("Elem # %d: %s" % (elemCount, line))
                elif search[0] == "Node":
                    nodeCount += 1
                    # print("Node # %d: %s" % (nodeCount, line))
                elif search[0] == "NodeString":
                    nodeStringCount += 1

        srhgeomfile.close()

        self.numOfElements = elemCount
        self.numOfNodes = nodeCount
        self.numOfNodeStrings = nodeStringCount

        print("There are %d elements, %d nodes, and %d node strings in the mesh." % (self.numOfElements,
                                                                                    self.numOfNodes, self.numOfNodeStrings))


    def readSRHGEOMFile(self):
        """ Get mesh information by reading the srhgeom file

        Parameters
        ----------

        Returns
        -------

        """

        print("Reading the SRHGEOM file ...")

        # read the "srhgeom" mesh file
        try:
            srhgeomfile = open(self.srhgeom_filename, 'r')
        except:
            print('Failed openning srhgeom file', self.srhgeom_filename)
            sys.exit()

        count = 0
        elemCount = 0
        nodeCount = 0
        nodeStringCound = 0

        while True:
            count += 1

            # Get next line from file
            line = srhgeomfile.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break

            # print("Line{}: {}".format(count, line.strip()))

            search = line.split()
            # print(search)

            if len(search) != 0:
                if search[0] == "Elem":
                    elemCount += 1
                    # print("Elem # %d: %s" % (elemCount, line))
                    self.elementNodesList[elemCount - 1][0:len(search[2:])] = search[2:]
                    self.elementNodesCount[elemCount - 1] = len(search[2:])
                    if len(search[2:]) < 1 or len(search[2:]) > gMax_Nodes_per_Element:
                        sys.exit("Number of nodes for element %d is less than 1 or larger than the max of %d." % (elemCount,gMax_Nodes_per_Element))
                    self.vtkCellTypeCode[elemCount - 1] = vtkCellTypeMap[len(search[2:])]
                elif search[0] == "Node":
                    nodeCount += 1
                    # print("Node # %d: %s" % (nodeCount, line))
                    self.nodeCoordinates[nodeCount - 1] = search[2:]
                elif search[0] == "NodeString":
                    nodeStringCound += 1
                    self.nodeStringsDict[search[1]] = search[2:]

        srhgeomfile.close()

        #calculate the bed elevation at cell center
        for cellI in range(self.numOfElements):

            elev_temp = 0.0

            #loop over all nodes of current element
            for nodeI in range(self.elementNodesCount[cellI]):
                elev_temp += self.nodeCoordinates[self.elementNodesList[cellI][nodeI]-1,2]  #z elevation; nodeI-1 because node number is 1-based for SRH-2D

            self.elementBedElevation[cellI] = elev_temp / self.elementNodesCount[cellI]

        if False:
            print("elementNodesList = ", self.elementNodesList)
            print("elementNodesCount = ", self.elementNodesCount)
            print("vtkCellTypeCode = ", self.vtkCellTypeCode)
            print("nodeCoordinates = ", self.nodeCoordinates)
            print("elementBedElevation = ", self.elementBedElevation)
            print("nodeStrings = ", self.nodeStringsDict)

    def buildNodeElements(self):
        """ Build node's element list for all nodes

        Returns
        -------

        """

        #create an emptp list of lists for all nodes
        self.nodeElementsList = [[] for _ in range(self.numOfNodes)]

        #loop over all cells
        for cellI in range(self.numOfElements):
            #loop over all nodes of current cell
            for i in range(self.elementNodesCount[cellI]):
                #get the node ID of current node
                nodeID = self.elementNodesList[cellI][i]

                self.nodeElementsList[nodeID-1].append(cellI)  #nodeID-1 because SRH-2D is 1-based

        #count beans in each node's element list
        for nodeI in range(self.numOfNodes):
            self.nodeElementsCount[nodeI] = len(self.nodeElementsList[nodeI])

        #print("nodeElementsCount = ", self.nodeElementsCount)
        #print("nodeElementsList = ", self.nodeElementsList)

class SRH_2D_SRHMat:
    """A class to handle srhmat file for SRH-2D

    Attributes:


    Methods:

    """
    def __init__(self, srhmat_filename):
        self.srhmat_filename = srhmat_filename

        #number of materials (Manning's n zones)
        self.numOfMaterials = -1

        #list of Material names (in a dictionary: ID and name)
        self.matNameList = {}

        #each material zone's cell list (in a dictionary: material zone ID and cell list)
        self.matZoneCells = {}

        #build material zones data
        self.buildMaterialZonesData()

    def buildMaterialZonesData(self):
        """Build the data for material zones

        Returns
        -------

        """
        print("Reading the SRHMAT file ...")

        # read the "srhmat" material file
        try:
            srhmatfile = open(self.srhmat_filename, 'r')
        except:
            print('Failed openning SRHMAT file', self.srhmat_filename)
            sys.exit()

        res_MatNames = {} #dict to hold "MatName" entries for material names list: ID and name
        res_Materials = {} #dict to hold "Material" information: ID and list of cells

        current_MaterialID = -1
        current_MaterialCellList = []

        while True:
            # Get a line from file
            line = srhmatfile.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break

            # print("Line{}: {}".format(count, line.strip()))

            search = line.split()
            # print(search)

#            print("search[0] == Material", (search[0] == "Material"))

            if len(search) != 0: #if it is not just an empty line
                if search[0] == "SRHMAT":
                    continue
                elif search[0] == "NMaterials":
                    self.numOfMaterials = int(search[1])
                elif search[0] == "MatName":
                    res_MatNames[search[1]] = search[2]
                elif search[0] == "Material": # a new material zone starts
                    #if this is not the first Material zone; save the previous Material zone
                    if ((current_MaterialID != -1) and (len(current_MaterialCellList)!=0)):
                        res_Materials[current_MaterialID] = copy.deepcopy(current_MaterialCellList)

                        #clear up current_MaterialID and current_MaterialCellList
                        current_MaterialID = -1
                        current_MaterialCellList.clear()

                    current_MaterialID = int(search[1])
                    current_MaterialCellList.extend(search[2:])
                else: #still in a material zone (assume there is no other things other than those listed above in the
                      #srhmat file)
                    current_MaterialCellList.extend(search)

            #add the last material zone
            res_Materials[current_MaterialID] = copy.deepcopy(current_MaterialCellList)

        srhmatfile.close()

        #convert material ID from str to int
        for zoneID, cellList in res_Materials.items():
            print("cellList ", cellList)
            temp_list = [int(i) for i in cellList]
            res_Materials[zoneID] = temp_list

        self.matNameList = res_MatNames
        self.matZoneCells = res_Materials

        #print("matNameList = ", self.matNameList)
        #print("matZoneCells = ", self.matZoneCells)


    def find_cell_material_ID(self, cellID):
        """ Given a cell ID, find its material (Manning's n) ID

        Parameters
        ----------
        cellID:

        Returns
        -------

        """

        if not self.matZoneCells:
            self.buildMaterialZonesData()

        #whether the cell is found in the list
        bFound = False

        for matID, cellList in self.matZoneCells.items():
            if (cellID+1) in cellList:  #cellID+1 because SRH-2D is 1-based
                bFound = True
                return matID

        #if cellID is not found, report problem
        if not bFound:
            print("In find_cell_material_ID(cellID), cellID =", cellID, "is not found. Check mesh. Exiting ...")
            sys.exit()


class SRH_2D_Data:
    """
    A class for SRH-2D data I/O, manipulation, and format conversion
    
    This class is designed to read SRH-2D results in format other than VTK. It can 
    save SRH-2D results into VTK format for visualization in Paraview, parse
    SRH-2D mesh information and convert/save to other formats, query SRH-2D
    results (e.g., for calibration), etc.
    
    Attributes
    ----------
    hdf_filename : str
        the name for the HDF result file generated by HEC-RAS

    srhhydro_obj: {SRH_2D_SRHHydro} -- an object to hold information in the SRHHYDRO file
    srhgeom_obj: {SRH_2D_Geom} -- an object to hold information in the SRHGEOM file
    srhmat_obj: {SRH_2D_Mat} -- an object to hold information in the SRHMAT file
    
    
    Methods
    -------
    get_units()    
        Get the units used in the SRH-2D project
    
    """
    
    def __init__(self, srhhydro_filename, srhgeom_filename, srhmat_filename):
        self.srhhydro_filename = srhhydro_filename
        self.srhgeom_filename = srhgeom_filename
        self.srhmat_filename = srhmat_filename

        #read and build SRH_2D_SRHHydro, SRH_2D_Geom, and SRH_2D_Mat objects
        self.srhhydro_obj = SRH_2D_SRHHydro(self.srhhydro_filename)
        self.srhgeom_obj = SRH_2D_SRHGeom(self.srhgeom_filename)
        self.srhmat_obj = SRH_2D_SRHMat(self.srhmat_filename)

        #Manning's n value at cell centers and nodes
        self.ManningN_cell = np.zeros(self.srhgeom_obj.numOfElements, dtype=float)
        self.ManningN_node = np.zeros(self.srhgeom_obj.numOfNodes, dtype=float)

        #build Manning's n value at cell centers and nodes
        self.buildManningN_cells_nodes()

        #XMDF data: if desired, the result data in a XMDF file
        #can be read in and stored in the following arrays.
        #Depending on whether the XMDF is nodal or at cell centers, use different arrays.
        self.xmdfTimeArray_Nodal = None   #numpy array to store all time values
        self.xmdfAllData_Nodal = {}     #dict to store {varName: varValues (numpy array)}

        self.xmdfTimeArray_Cell = None   #numpy array to store all time values
        self.xmdfAllData_Cell = {}     #dict to store {varName: varValues (numpy array)}

    def get_case_name(self):
        """Get the "Case" name for this current case ("Case" in srhhyro file)

        Returns
        -------
        case_name: {string} -- case name

        """
        return self.srhhydro_obj.srhhydro_content["Case"]

    def buildManningN_cells_nodes(self):
        """ Build Manning's n values at cell centers and nodes

        This calculation is based on the srhhydro and srhgeom files, not interpolation from a GeoTiff file.

        Returns
        -------

        """

        print("Building Manning's n values for cells and nodes in SRH-2D mesh ...")

        #Manning's n dictionary in srhhydro
        nDict = self.srhhydro_obj.srhhydro_content["ManningsN"]
        print("nDict = ", nDict)

        #loop over all cells in the mesh
        for cellI in range(self.srhgeom_obj.numOfElements):
            #get the material ID of current cell
            matID = self.srhmat_obj.find_cell_material_ID(cellI)

            #print("matID = ", matID)
            #print("nDict[matID] = ", nDict[matID])

            #get the Manning's n value for current cell
            self.ManningN_cell[cellI] = nDict[matID]

        #interpolate Manning's n from cell centers to nodes



    def readSRHXMDFFile(self, xmdfFileName, bNodal):
        """ Read SRH-2D result file in XMDF format (current version 13.1.6 of SMS only support data at node).

        Attributes:
        xmdfFileName: the file name for the XMDF data
        bNodal: bool, whether it is nodal (True) or at cell center (False)

        Returns
        -------
        variable names, variable data

        """

        #for debug
        dumpXMDFFileItems(xmdfFileName)

        if bNodal:
            if "XMDFC" in xmdfFileName:
                print("Warning: bNodal indices the result is for nodal values. However, the XMDF file name contains "
                      "\"XMDFC\", which indicates it is for cell center values. Please double check.")
        else:
            if "XMDFC" not in xmdfFileName:
                print("Warning: bNodal indices the result is for cell centers. However, the XMDF file name "
                      "does not contain \"XMDFC\", which indicates it is for nodal values. Please double check.")

        print("Reading the XMDF file ...\n")

        xmdfFile = h5py.File(xmdfFileName, "r")

        #build the list of solution variables
        varNameList = []

        #copy of the original velocity vector variable name
        varNameVelocity = ''

        for ds in xmdfFile.keys():
            #print(ds)

            if ds != "File Type" and ds != "File Version":
                if "Velocity" in ds:
                    varNameVelocity = '%s' % ds
                    vel_x = ds.replace("Velocity","Vel_X",1)
                    vel_y = ds.replace("Velocity","Vel_Y",1)
                    varNameList.append(vel_x)
                    varNameList.append(vel_y)
                else:
                    varNameList.append(ds)

        if not varNameList: #empty list
            print("There is no solution varialbes in the XMDF file. Exiting ...")
            sys.exit()

        print("Variables in the XMDF file: ", varNameList)

        #build the 1d array of time for the solution
        #It seems all solution variables share the same time (although
        #each has their own recrod of "Time"). So only get one is sufficient.
        if bNodal:
            self.xmdfTimeArray_Nodal = np.array(xmdfFile[varNameList[0]]['Times'])
            print("Time values for the solutions: ", self.xmdfTimeArray_Nodal)
        else:
            self.xmdfTimeArray_Cell = np.array(xmdfFile[varNameList[0]]['Times'])
            print("Time values for the solutions: ", self.xmdfTimeArray_Cell)

        #result for all varialbes and at all times in a dictionary (varName : numpy array)
        for varName in varNameList:
            print("varName =", varName)

            varName_temp = varName

            if ("Vel_X" in varName) or ("Vel_Y" in varName):  # if it is velocity components
                varName = varNameVelocity                     # use the original velocity variable name

            if bNodal: #for nodal data
                if "Velocity" in varName:
                    if "Vel_X" in varName_temp: #vel-x
                        #print("varName_temp = ", varName_temp)
                        self.xmdfAllData_Nodal[varName_temp] = np.array(xmdfFile[varName]['Values'][:,:,0])
                    else:
                        #print("varName_temp = ", varName_temp)
                        self.xmdfAllData_Nodal[varName_temp] = np.array(xmdfFile[varName]['Values'][:,:,1])
                else:
                    self.xmdfAllData_Nodal[varName] = np.array(xmdfFile[varName]['Values'])

                    #fixe the water elevation = -999 in SRH-2D
                    if varName == "Water_Elev_ft" or varName == "Water_Elev_m":
                        for nodeI in range(len(self.xmdfAllData_Nodal[varName])):
                            if self.xmdfAllData_Nodal[varName][nodeI] == -999:
                                self.xmdfAllData_Nodal[varName][nodeI] = self.srhgeom_obj.nodeCoordinates[nodeI, 2] #use node elevation


                if np.array(xmdfFile[varName]['Values']).shape[1] != self.srhgeom_obj.numOfNodes:
                    print("The number of nodes in the XMDF file (%d) is different from that in the mesh (%d). "
                          "Abort readSRHXMDFFile(...) function."
                          % (np.array(xmdfFile[varName]['Values']).shape[1], self.srhgeom_obj.numOfNodes))
                    return

                print(self.xmdfAllData_Nodal)
            else: #for cell center data
                if "Velocity" in varName:
                    if "Vel_X" in varName_temp: #vel-x
                        #print("varName_temp, varName = ", varName_temp, varName)
                        self.xmdfAllData_Cell[varName_temp] = np.array(xmdfFile[varName]['Values'])[:,:,0]
                    else: #vel-y
                        #print("varName_temp = ", varName_temp)
                        self.xmdfAllData_Cell[varName_temp] = np.array(xmdfFile[varName]['Values'])[:,:,1]
                else:
                    self.xmdfAllData_Cell[varName] = np.array(xmdfFile[varName]['Values'])

                    #fixe the water elevation = -999 in SRH-2D
                    if varName == "Water_Elev_ft" or varName == "Water_Elev_m":
                        #loop through time
                        for timeI in range(self.xmdfAllData_Cell[varName].shape[0]):
                            for cellI in range(self.xmdfAllData_Cell[varName].shape[1]):
                                if self.xmdfAllData_Cell[varName][timeI, cellI] == -999:
                                    self.xmdfAllData_Cell[varName][timeI, cellI] = self.srhgeom_obj.elementBedElevation[cellI]

                if np.array(xmdfFile[varName]['Values']).shape[1] != self.srhgeom_obj.numOfElements:
                    print("The number of elements in the XMDF file (%d) is different from that in the mesh (%d). "
                          "Abort readSRHXMDFFile(...) function."
                          % (np.array(xmdfFile[varName]['Values']).shape[1], self.srhgeom_obj.numOfElements))

                print(self.xmdfAllData_Cell)

        xmdfFile.close()

    def outputXMDFDataToVTK(self, bNodal, timeStep=-1,lastTimeStep=False, dir=''):
        """Output XMDF result data to VTK

        This has to be called after the XMDF data have been loaded by calling readSRHXMDFFile(...)

        call outputVTK(self, vtkFileName, resultVarNames, resultData, bCellData)

        Attributes:
          bNodal: {bool} -- whether export nodal data or cell center data. Currently, it can't output both.
          timeStep: {int} -- optionly only the specified time step will be saved
          lastTimeStep: {bool} -- optionally specify only the last time step
          dir: {string} -- optional directory

        Returns
        -------

        """
        print("Output all data in the XMDF file to VTK ...")

        if (bNodal and (not self.xmdfAllData_Nodal)) or ((not bNodal) and (not self.xmdfAllData_Cell)):
            print("Empty XMDF data arrays. Call readSRHXMDFFile() function first. Exiting ...")
            sys.exit()

        #check the sanity of timeStep
        if (timeStep != -1):
            if bNodal:
                if (not timeStep in range(len(self.xmdfTimeArray_Nodal))):
                    message = "Specified timeStep = %d not in range (0 to %d)." % (timeStep, len(self.xmdfTimeArray_Nodal))
                    sys.exit(message)
            else:
                if (not timeStep in range(len(self.xmdfTimeArray_Cell))):
                    message = "Specified timeStep = %d not in range (0 to %d)." % (timeStep, len(self.xmdfTimeArray_Cell))
                    sys.exit(message)

        vtkFileName_base = ''
        print("self.srhhydro_obj.srhhydro_content[\"Case\"] = ", self.srhhydro_obj.srhhydro_content["Case"])

        if dir!='':
            vtkFileName_base = dir + '/' + 'SRH2D_' + self.srhhydro_obj.srhhydro_content["Case"] #use the case name of the
            # SRH-2D case
        else:
            vtkFileName_base = 'SRH2D_' + self.srhhydro_obj.srhhydro_content["Case"]  # use the case name of the SRH-2D case
        print("vtkFileName_base = ", vtkFileName_base)

        #build result variable names
        resultVarNames = list(self.xmdfAllData_Nodal.keys()) if bNodal else list(self.xmdfAllData_Cell.keys())

        #add bed elevation
        #get the units of the results
        units = self.srhhydro_obj.srhhydro_content['OutputFormat'][1]
        print("SRH-2D result units:", units)
        if units == "SI":
            resultVarNames.append("Bed_Elev_m")
        else:
            resultVarNames.append("Bed_Elev_ft")

        print("All solution variable names: ", resultVarNames)

        #add Manning's n
        resultVarNames.append("ManningN")

        #loop through each time step
        timeArray = self.xmdfTimeArray_Nodal if bNodal else self.xmdfTimeArray_Cell
        for timeI in range(timeArray.shape[0]):

            if lastTimeStep:
                if timeI < (timeArray.shape[0] - 1):
                    continue

            if (timeStep != -1) and (timeI != timeStep):
                continue

            print("timeI and time = ", timeI, timeArray[timeI])

            #numpy array for all solution variables at one time
            resultData =      np.zeros((self.srhgeom_obj.numOfNodes, len(resultVarNames)), dtype="float32") if bNodal \
                         else np.zeros((self.srhgeom_obj.numOfElements, len(resultVarNames)), dtype="float32")

            str_extra = "_N_" if bNodal else "_C_"

            vtkFileName = vtkFileName_base + str_extra + str(timeI).zfill(4) + ".vtk"
            print("vtkFileName = ", vtkFileName)

            #loop through each solution variable
            for varName, varI in zip(resultVarNames, range(len(resultVarNames)-1)):
                print("varName = ", varName)
                if ("Bed_Elev" not in varName) and ("ManningN" not in varName):
                    #get the values of current solution varialbe at current time
                    resultData[:,varI] =      self.xmdfAllData_Nodal[varName][timeI,:] if bNodal \
                                         else self.xmdfAllData_Cell[varName][timeI,:]

            #add the bed elevation (the last one in resultData)
            resultData[:,len(resultVarNames)-2] = self.srhgeom_obj.nodeCoordinates[:,2] if bNodal \
                                     else self.srhgeom_obj.elementBedElevation

            #add Mannng's n
            resultData[:, len(resultVarNames)-1] = self.ManningN_node if bNodal \
                                    else self.ManningN_cell

            #call the vtk output function
            self.outputVTK(vtkFileName, resultVarNames, resultData, bNodal)


    def readSRHFile(self, srhFileName):
        """ Read SRH-2D result file in SRHC (cell center) or SRH (point) format.

        Note: SRH-2D outputs an extra "," to each line. As a result, Numpy's
        genfromtext(...) function adds a column of "nan" to the end.

        Returns
        -------
        variable names, variable data

        """

        print("Reading the SRH/SRHC result file ...")

        data = np.genfromtxt(srhFileName, delimiter=',', names=True)

        return data.dtype.names[:-1], data

    def readTECFile(self, tecFileName):
        """Read SRH-2D results in Tecplot format

        Parameters
        ----------
        tecFileName: Tecplot file name

        Returns
        -------

        """

        readerTEC = vtk.vtkTecplotReader()
        readerTEC.SetFileName(tecFileName)
        # 'update' the reader i.e. read the Tecplot data file
        readerTEC.Update()

        polydata = readerTEC.GetOutput()

        # print(polydata)  #this is a multibock dataset
        # print(polydata.GetBlock(0))  #we only need the first block

        # If there are no points in 'vtkPolyData' something went wrong
        if polydata.GetBlock(0).GetNumberOfPoints() == 0:
            raise ValueError(
                "No point data could be loaded from '" + tecFileName)
            return None

        #return of the first block (there should be only one block)
        return polydata.GetBlock(0)


    def outputVTK(self, vtkFileName, resultVarNames, resultData, bNodal):
        """ Output result to VTK file

        The supplied resultVarNames and resultData should be compatible with the mesh. If resultVarNames is empty,
        it only outputs the mesh with no data.

        Parameters
        ----------
        vtkFileName: name for the output vtk file
        resultVarNames: result variable names
        resultData: result data
        bNodal: whether the data is nodal (True) or at cell center (False)


        Returns
        -------

        """

        print("Output to VTK ...")

        try:
            fid = open(vtkFileName, 'w')
        except IOError:
            print('vtk file open error')
            sys.exit()

        #get the units of the results
        units = self.srhhydro_obj.srhhydro_content['OutputFormat'][1]
        print("SRH-2D result units:", units)

        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('Results from SRH-2D Modeling Run\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        fid.write('\n')

        # output points
        fid.write('POINTS %d double\n' % self.srhgeom_obj.nodeCoordinates.shape[0])

        point_id = 0  # point ID counter
        for k in range(self.srhgeom_obj.nodeCoordinates.shape[0]):
            point_id += 1
            fid.write(" ".join(map(str, self.srhgeom_obj.nodeCoordinates[k])))
            fid.write("\n")

        # output elements
        fid.write('CELLS %d %d \n' % (self.srhgeom_obj.elementNodesList.shape[0],
                                      self.srhgeom_obj.elementNodesList.shape[0] + np.sum(self.srhgeom_obj.elementNodesCount)))

        cell_id = 0  # cell ID counter
        for k in range(self.srhgeom_obj.elementNodesList.shape[0]):
            cell_id += 1
            fid.write('%d ' % self.srhgeom_obj.elementNodesCount[k])
            fid.write(" ".join(map(str, self.srhgeom_obj.elementNodesList[k][:self.srhgeom_obj.elementNodesCount[k]] - 1)))
            fid.write("\n")

        # output element types
        fid.write('CELL_TYPES %d \n' % self.srhgeom_obj.elementNodesList.shape[0])

        for k in range(self.srhgeom_obj.elementNodesList.shape[0]):
            fid.write('%d ' % self.srhgeom_obj.vtkCellTypeCode[k])
            if (((k + 1) % 20) == 0):
                fid.write('\n')

        fid.write('\n')

        # How many solution data entries to output depending on whehter it is nodal or not
        entryCount = -1
        if bNodal:
            entryCount = self.srhgeom_obj.nodeCoordinates.shape[0]
        else:
            entryCount = self.srhgeom_obj.elementNodesList.shape[0]

        # output solution variables: only if there is solution variable
        if len(resultVarNames) != 0:
            if not bNodal:
                print('Solution variables are at cell centers. \n')
                fid.write('CELL_DATA %d\n' % self.srhgeom_obj.elementNodesList.shape[0])
            else:
                print('Solution variables are at vertices. \n')
                fid.write('POINT_DATA %d\n' % self.srhgeom_obj.nodeCoordinates.shape[0])

            # column numbers for Vel_X and Vel_Y for vector assemble
            nColVel_X = -1
            nColVel_Y = -1

            # First output all solution variables as scalars
            print('The following solution variables are processed: \n')
            for k in range(len(resultVarNames)):
                print('     %s\n' % resultVarNames[k])

                if resultVarNames[k].find('Vel_X') != -1:
                    nColVel_X = k
                elif resultVarNames[k].find('Vel_Y') != -1:
                    nColVel_Y = k

                fid.write('SCALARS %s double 1 \n' % resultVarNames[k])
                fid.write('LOOKUP_TABLE default\n')

                for entryI in range(entryCount):
                    fid.write('%f ' % resultData[entryI][k])

                    if (((entryI + 1) % 20) == 0):
                        fid.write('\n')

                fid.write('\n \n')

            # Then output Vel_X and Vel_Y as velocity vector (Vel_Z = 0.0)
            # print('nColVel_X, nColVel_Y = %d, %d' % (nColVel_X, nColVel_Y))
            if (nColVel_X != -1) and (nColVel_Y != -1):
                if units == "SI":
                    fid.write('VECTORS Velocity_m_p_s double \n')
                else:
                    fid.write('VECTORS Velocity_ft_p_s double \n')

                for entryI in range(entryCount):
                    fid.write('%f %f 0.0   ' % (resultData[entryI][nColVel_X],
                                                resultData[entryI][nColVel_Y]))

                    if (((entryI + 1) % 20) == 0):
                        fid.write('\n')

                fid.write('\n \n')

        fid.close()

    def cell_center_to_vertex(self, cell_data, vertex_data):
        """Interpolate result from cell center to vertex

        Returns
        -------

        """
        pass

    def vertex_to_cell_center(self, vertex_data, cell_data):
        """Interpolate result from vertex to cell center

        Returns
        -------

        """
        pass


def main():
    """ Testing SRH_2D_data class

    Returns
    -------

    """

    my_srh_2d_data = SRH_2D_Data("Muncie2D.srhhydro", "Muncie2D.srhgeom", "Muncie2D.srhmat")

    # User specified SRH-2D result in SRH (point) or SRHC (cell center) format
    srhFileName = 'Muncie2D_SRHC2.dat'

    # whehter the data is nodal (True) or at cell center (False) (need to be correctly set by user)
    bNodal = True

    # Read SRH-2D result file
    resultVarNames, resultData = my_srh_2d_data.readSRHFile(srhFileName)
    # print(resultVarNames, resultData)

    # output SRH-2D result to VTK
    srhName = os.path.splitext(srhFileName)[0]
    vtkFileName = srhName + ".vtk"
    print("vtk file name = ", vtkFileName)

    # Note: in resultData, the 1st, second, third, and fourth columns are Point_ID,
    #      X_unit, Y_unit, and Bed_Elev_unit respectively. There is no need to save them
    #      in vtk.
    my_srh_2d_data.outputVTK(vtkFileName, resultVarNames, resultData[:,4:], bNodal)

    print("All done!")


if __name__ == "__main__":
    main()