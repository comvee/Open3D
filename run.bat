@echo off
set config_path=%1

python examples/python/reconstruction_system/run_system.py --config %config_path% --make
python examples/python/reconstruction_system/run_system.py --config %config_path% --register
python examples/python/reconstruction_system/run_system.py --config %config_path% --refine
python examples/python/reconstruction_system/run_system.py --config %config_path% --integrate
python examples/python/reconstruction_system/construct_point_cloud.py --config %config_path%