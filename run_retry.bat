@echo off
set config_path=%1

python examples/python/reconstruction_system/run_system.py --config %config_path% --register
timeout /t 1
python examples/python/reconstruction_system/run_system.py --config %config_path% --refine
timeout /t 1
python examples/python/reconstruction_system/run_system.py --config %config_path% --integrate