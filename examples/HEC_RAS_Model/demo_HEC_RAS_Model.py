"""
Test the HEC_RAS_Model class

It demonstrates how to use pyHMT2D to control the run of HEC-RAS model.
"""

#if run in the cloud, need to specify the location of pyHMT2D. If pyHMT2D is installed
#with regular pip install, then the following is not necessary.
# import sys
# sys.path.append(r"C:\Users\Administrator\python_packages\pyHMT2D")

import sys
#sys.path.append((r"C:\Users\xzl123\Dropbox\PycharmProjects\pyHMT2D"))

#print('\n'.join(sys.path))

import pyHMT2D
import os
import math
import shutil
# import geopandas as gpd
import osgeo
from osgeo import gdal, ogr
import glob
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fiona
from shapely.geometry import box
import geopandas as gpd
import rasterio.mask
import pycrs

os.environ['GDAL_DATA'] = r'c:\users\fzr5082\anaconda3\envs\pyhmt\lib\site-packages\osgeo\gdal-data'
os.environ['PROJ_LIB'] = r'C:\Users\fzr5082\Anaconda3\envs\pyhmt\Library\share\proj'

# from examples.RAS_2D_Monte_Carlo.Munice2D_ManningN_Monte_Carlo.demo_HEC_RAS_Monte_Carlo import postprocess_results
def run_HEC_RAS():

    #create a HEC-RAS model instance
    #my_hec_ras_model = pyHMT2D.RAS_2D.HEC_RAS_Model(version="5.0.7",faceless=False)
    #my_hec_ras_model = pyHMT2D.RAS_2D.HEC_RAS_Model(version="6.0.0", faceless=False)
    my_hec_ras_model = pyHMT2D.RAS_2D.HEC_RAS_Model(version="6.1.0", faceless=False)

    #initialize the HEC-RAS model
    my_hec_ras_model.init_model()

    print("Hydraulic model name: ", my_hec_ras_model.getName())
    print("Hydraulic model version: ", my_hec_ras_model.getVersion())

    #open a HEC-RAS project
    # my_hec_ras_model.open_project("Muncie2D.prj", "Terrain/TerrainMuncie_composite.tif")
    project = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\spring_pro2_far.prj"
    terrain = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\Terrain\USGS_13_n41w078_20190417_pro.tif"
    my_hec_ras_model.open_project(project, terrain)

    #run the HEC-RAS model's current project
    my_hec_ras_model.run_model()

    #close the HEC-RAS project
    my_hec_ras_model.close_project()

    #quit HEC-RAS
    my_hec_ras_model.exit_model()

def convert_HEC_RAS_to_VTK():
    """ Convert HEC-RAS result to VTK

    Returns
    -------

    """

    # my_ras_2d_data = pyHMT2D.RAS_2D.RAS_2D_Data("Muncie2D.p01.hdf",
    #                                      "Terrain/TerrainMuncie_composite.tif")

    project = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\spring_pro2_far.p01.hdf"
    terrain = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\Terrain\USGS_13_n41w078_20190417_pro.tif"
    my_ras_2d_data = pyHMT2D.RAS_2D.RAS_2D_Data(project,
                                                terrain)

    my_ras_2d_data.saveHEC_RAS2D_results_to_VTK(lastTimeStep=True)

def change_flow(flow):
    f = open(r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\spring_pro2_far.u01", "r")
    data = f.readlines()
    f.close()
    file_final = open(r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\spring_pro2_far.u01", "w")
    key = 'Flow Hydrograph='
    for line_number, line in enumerate(data):
        if key in line:
            # to get the number of hours
            num_hours = int(data[line_number][len(key): len(key) + 4])
            i = 0
            count = math.floor(num_hours/10)
            flow_str = "     " + str(flow).zfill(3)
            for i in range(count):
                data[line_number + 1 + i] = 10 * flow_str + '\n'
            if (num_hours % 10 != 0):
                data[line_number + count + 1] = (num_hours % 10) * flow_str + '\n'
    file_final.writelines(data)
    file_final.close()

def post_processing_copy_results(flow):
    src = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\Plan p01"
    p0 = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python\WSE_depth"
    dest = os.path.join(os.path.sep, p0, str(flow))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

def determine_flood_area(flow,  root):
    # getting the spatial reference of the DEM file
    p_terrain = os.path.join(os.path.sep, root, "Terrain", "USGS_13_n41w078_20190417_pro.tif")
    terrain = gdal.Open(p_terrain)
    srs =osgeo.osr.SpatialReference()
    main_path = os.path.join(os.path.sep, root, "WSE_depth", str(flow))
    WSE_tif_path = glob.glob(main_path + "\\WSE*.tif")
    depth_tif_path = glob.glob(main_path + "\\Depth*.tif")
    hand_tif_path = os.path.join(os.path.sep, root, "Terrain", "hand.tif")
    WSE = gdal.Open(WSE_tif_path[0])
    depth = gdal.Open(depth_tif_path[0])
    hand = gdal.Open(hand_tif_path)
    dest_dir = os.path.join(os.path.sep, root, "flood_Area", str(flow))
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    rs_lst= []
    rs_lst.append(WSE_tif_path[0])
    rs_lst.append(depth_tif_path[0])
    rs_lst.append(hand_tif_path)
    out_names = ["WSE_clip.tif", "depth_clip.tif", 'hand_clip.tif']

    # clipy
    territory_shp = os.path.join(os.path.sep, root, "Terrain", "territory.shp")
    with fiona.open(territory_shp, "r") as shpfile:
        shapes = [feature["geometry"] for feature in shpfile]
        # read raster files one by one
        for i, rs in enumerate(rs_lst):
            with rasterio.open(rs) as src:
                bbox = box(np.maximum(src.bounds[0], shpfile.bounds[0]),  # left
                           np.maximum(src.bounds[1], shpfile.bounds[1]),  # bottom
                           np.minimum(src.bounds[2], shpfile.bounds[2]),  # right
                           np.minimum(src.bounds[3], shpfile.bounds[3])  # top
                           )
                geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:26917")
                geo = geo.to_crs(crs=src.crs.data['init'])
                coords = getFeatures(geo)
                out_img, out_transform = rasterio.mask.mask(dataset=src, shapes=coords, crop=True)
                out_meta = src.meta.copy()
                # we need to parse the epsg value from the CRS so that we can create a proj4 -string using pyCRS
                # library (to ensure that the projection information is saved correctly)
                epsg_code = int(src.crs.data['init'][5:])
                # print(epsg_code)
                out_meta.update({"driver": "GTiff",
                                 "height": out_img.shape[1],
                                 "width": out_img.shape[2],
                                 "transform": out_transform,
                                 # "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                                 # "crs": rasterio.crs.CRS.from_epsg(26917)}
                                 "crs": rasterio.crs.CRS({"init": "epsg:26917"})}
                                )
                out_tif = os.path.join(dest_dir, out_names[i])

                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(out_img)
                # clipped = rasterio.open(out_tif)
                # show(clipped, cmap='terrain')
                dest.close()
                ### to release the memory
                dest = None
                out_img = None


    gdal.UseExceptions()


    # residence area shape file
    res_shp_path = os.path.join(os.path.sep, root, "Terrain", "residence", "residence.shp")


    # cliping WSE_clip with residence shape files
    out_WSE_res = os.path.join(os.path.sep, dest_dir, "WSE_clip_res.tif")
    # ras_clip_shp needs (input_raster path, shapefile, output_raster path
    ras_clip_shp(os.path.join(os.path.sep, dest_dir, "WSE_clip.tif"),
                 res_shp_path,
                 out_WSE_res)

    out_depth_res = os.path.join(os.path.sep, dest_dir, "depth_clip_res.tif")
    ras_clip_shp(os.path.join(os.path.sep, dest_dir, "depth_clip.tif"),
                 res_shp_path,
                 out_depth_res)

    out_hand_res = os.path.join(os.path.sep, dest_dir, "hand_clip_res.tif")
    ras_clip_shp(os.path.join(os.path.sep, dest_dir, "hand_clip.tif"),
                 res_shp_path,
                 out_hand_res)

    WSE_clip_res = gdal.Open(out_WSE_res)
    depth_clip_res = gdal.Open(out_depth_res)
    hand_clip_res = gdal.Open(out_hand_res)
    xx_WSE, yy_WSE = get_lat_lon(WSE_clip_res)
    WSE_arr = WSE_clip_res.GetRasterBand(1).ReadAsArray()

    xx_depth, yy_depth = get_lat_lon(depth_clip_res)
    depth_arr = depth_clip_res.GetRasterBand(1).ReadAsArray()

    xx_hand, yy_hand = get_lat_lon(hand_clip_res)
    hand_arr = hand_clip_res.GetRasterBand(1).ReadAsArray()

    flood_arr = np.ones((5, depth_arr.shape[0], depth_arr.shape[1]))
    flood_arr[0, :, :] = np.where(depth_arr < 0, -9999, depth_arr)
    flood_arr[1, :, :] = np.where(depth_arr < 0, -9999, WSE_arr)
    flood_arr[2, :, :] = hand_arr
    flood_arr[3, :, :] = xx_depth[:depth_arr.shape[0], : depth_arr.shape[1]]
    flood_arr[4, :, :] = yy_depth[:depth_arr.shape[0], : depth_arr.shape[1]]
    np.save(os.path.join(os.path.sep, dest_dir, "flood_arr.npy"), flood_arr)

    # make a csv file
    col = ["depth(m)", "WSE", "HAND", "lat", "lon"]
    flood_pd = pd.DataFrame(columns=col)
    flood_pd.columns = ["depth(m)", "WSE", "HAND", "lat", "lon"]
    flood_pd["depth(m)"] = depth_arr[depth_arr > 0]
    flood_pd["WSE"] = WSE_arr[depth_arr > 0]
    flood_pd["HAND"] = hand_arr[depth_arr > 0]
    flood_pd["lat"] = xx_depth[:depth_arr.shape[0], : depth_arr.shape[1]][depth_arr > 0]
    flood_pd["lon"] = yy_depth[:depth_arr.shape[0], : depth_arr.shape[1]][depth_arr > 0]
    flood_pd.to_feather(os.path.join(os.path.sep, dest_dir, str(flow) + ".feather"), compression="uncompressed")
    flood_pd.to_csv(os.path.join(os.path.sep, dest_dir, str(flow) + ".csv"))




def ras_clip_shp(rs_path, shp_path, out_tif):
    with fiona.open(shp_path, "r") as shpfile:
        shapes = [feature["geometry"] for feature in shpfile]

    with rasterio.open(rs_path) as src:
        # bbox = box(np.maximum(src.bounds[0], shpfile.bounds[0]),  # left
        #            np.maximum(src.bounds[1], shpfile.bounds[1]),  # bottom
        #            np.minimum(src.bounds[2], shpfile.bounds[2]),  # right
        #            np.minimum(src.bounds[3], shpfile.bounds[3])  # top
        #            )
        # geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:26917")
        # geo = geo.to_crs(crs=src.crs.data['init'])
        # coords = getFeatures(geo)
        # out_img, out_transform = rasterio.mask.mask(dataset=src, shapes=coords, crop=True)
        out_img, out_transform = rasterio.mask.mask(dataset=src, shapes=shapes, crop=True)
        out_meta = src.meta.copy()
        # we need to parse the epsg value from the CRS so that we can create a proj4 -string using pyCRS
        # library (to ensure that the projection information is saved correctly)
        # epsg_code = int(src.crs.data['init'][5:])
        # print(epsg_code)
        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         # "crs": pycrs.parse.from_epsg_code(26917).to_proj4()}
                         "crs": rasterio.crs.CRS({"init": "epsg:26917"})}
                        )

        with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)
        # clipped = rasterio.open(out_tif)
        # show(clipped, cmap='terrain')
        dest.close()
        ### to release the memory
        dest = None
        out_img = None

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def create_polygon(coords):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()

def write_shapefile(poly, out_shp):
    """
    https://gis.stackexchange.com/a/52708/8104
    """
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(out_shp)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(poly)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def get_lat_lon(raster):
    xy = raster.GetGeoTransform()
    x = raster.RasterXSize
    y = raster.RasterYSize
    lon_start = xy[0]
    lon_stop = x * xy[1] + xy[0]
    lon_step = xy[1]
    lat_start = xy[3]
    lat_stop = y * xy[5] + xy[3]
    lat_step = xy[5]
    lons = np.arange(lon_start, lon_stop, lon_step)
    lats = np.arange(lat_start, lat_stop, lat_step)
    xx, yy = np.meshgrid(lons, lats)
    return xx, yy


if __name__ == "__main__":
    root = r"G:\Farshid\HEC-RAS\HEC-RAS\2D\spring_pro2_python"
    flow_list = [20, 37, 55, 118]   # 161, 200, 250, 311,
    for flow in flow_list:
        change_flow(flow)
        run_HEC_RAS()
        post_processing_copy_results(flow)
        determine_flood_area(flow, root)
        print("flow " + str(flow) + " cms done")

    print("All done!")

