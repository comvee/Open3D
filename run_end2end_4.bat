@echo off
set mkv_path=%1
set output_dir_path=%2
set config_path=%output_dir_path%/config.json
set pcd_path=%output_dir_path%/scene/integrated.ply

echo %config_path%
echo %pcd_path%
python examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input %mkv_path% --output %output_dir_path%

python examples/python/reconstruction_system/filter_frames.py --config %config_path%
python examples/python/reconstruction_system/run_system.py --config %config_path% --make
python examples/python/reconstruction_system/run_system.py --config %config_path% --register
python examples/python/reconstruction_system/run_system.py --config %config_path% --refine
python examples/python/reconstruction_system/run_system.py --config %config_path% --integrate
if not exist %pcd_path% (
    python examples/python/reconstruction_system/run_system.py --config %config_path% --register
    python examples/python/reconstruction_system/run_system.py --config %config_path% --refine
    python examples/python/reconstruction_system/run_system.py --config %config_path% --integrate
)

python examples/python/reconstruction_system/construct_point_cloud.py --config %config_path% --view_num 4