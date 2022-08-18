config=$1

python examples/python/reconstruction_system/filter_frames.py --config $config
python examples/python/reconstruction_system/run_system.py --config $config --make
sleep 0.5
python examples/python/reconstruction_system/run_system.py --config $config --register
sleep 0.5
python examples/python/reconstruction_system/run_system.py --config $config --refine
sleep 0.5
python examples/python/reconstruction_system/run_system.py --config $config --integrate
sleep 0.5
python examples/python/reconstruction_system/construct_point_cloud.py --config $config
# python examples/python/reconstruction_system/run_system.py --config $config --slac
# python examples/python/reconstruction_system/run_system.py --config $config --slac_integrate