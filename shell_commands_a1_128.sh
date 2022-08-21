## Shell commands documentation

#Marc 
#srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

#backend = numpy
#strong scaling sized 64 * 64
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=numpy


#strong scaling sized 128*128

srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

#strong scaling sized 256*256

srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=numpy


#strong scaling sized 512*512

srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=numpy

# weak scaling 
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=numpy

srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=numpy

#backend = gt:cpu_ifirst
#strong scaling sized 64 * 64
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst


#strong scaling sized 128*128

srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

#strong scaling sized 256*256

srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst


#strong scaling sized 512*512

srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

# weak scaling 
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_ifirst

#backend = gt:cpu_kfirst
#strong scaling sized 64 * 64
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst


#strong scaling sized 128*128

srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

#strong scaling sized 256*256

srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst


#strong scaling sized 512*512

srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

# weak scaling 
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:cpu_kfirst

#backend = gt:gpu
#strong scaling sized 64 * 64
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

#strong scaling sized 128*128

srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu


#strong scaling sized 256*256

srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu



#strong scaling sized 512*512

srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu


# weak scaling 
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=gt:gpu

#backend = gt:cuda
#strong scaling sized 64 * 64
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --
backend=cuda

srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --
backend=cuda

srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --
backend=cuda

srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --
backend=cuda

#strong scaling sized 128*128

srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=cuda


#strong scaling sized 256*256

srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result True --backend=cuda



#strong scaling sized 512*512

srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=cuda

srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result True --backend=cuda


# weak scaling 
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result True --
backend=cuda

srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result True --backend=cuda
