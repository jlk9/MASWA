import glob
import subprocess
import sys
import os

def compile_c_files(cc, obj, cflags, source):
    for thing in obj:
        command = [cc, cflags, '-c', '-o', source + thing + '.o', source + thing + '.c']
        print(' '.join(command))
        subprocess.check_call(command)

def compile_mpi():
    cc = 'mpic++'
    obj = ['MASWA_Ke_layer', 'MASWA_test_cases', 'MASWA_Ke_halfspace', 'MASWA_stiffness_matrix', 'MASWA_theoretical_dispersion_curve', 'MASWA_misfit', 'MASWA_inversion', 'MASWA_main', 'MASWA_setVariables']
    cflags = '-I./include'
    exe = 'maswa_mpi'
    libs = ['-lm']

    compile_c_files(cc, obj, cflags, 'src/')

    command = [cc, '-o', exe] + ['src/' + thing + '.o' for thing in obj] + libs
    print(' '.join(command))
    subprocess.check_call(command)
    

def compile_cuda():
    cc = 'g++'
    obj = ['MASWA_main']
    cflags = '-I./include_cuda'
    exe = 'maswa_cuda'
    libs = ['-lm', '-lcudart', '-lcublas', '-L/usr/lib/cuda/lib64']
    source = 'src_cuda/'

    # First we compile the cuda files:
    cuda_obj = ['MASWA_theoretical_dispersion_curve', 'MASWA_misfit', 'MASWA_stiffness_matrix', 'MASWA_test_cases', 'MASWA_ke', 'MASWA_inversion', 'MASWA_setVariables']
    for thing in cuda_obj:
        cuda_command = ['nvcc', '-c', source + thing + '.cu', '-o', source + thing + '_cuda.o', cflags] + libs + ['--prec-div=true']
        print(' '.join(cuda_command))
        subprocess.check_call(cuda_command)

    # Then the c files:
    compile_c_files(cc, obj, cflags, source)

    # Then put it all together:
    command = [cc, '-o', exe] + [source + thing + '.o' for thing in obj] + [source + thing + '_cuda.o' for thing in cuda_obj] + [cflags] + libs
    #command = [source + thing + '_cuda.o' for thing in cuda_obj] + [cflags] + libs
    print(' '.join(command))
    subprocess.check_call(command)

def clean(source):
    files = glob.glob(source + '*.o')
    for f in files:
        os.remove(f)

def main():
    if len(sys.argv) == 1:
        print('Error: specify either mpi or cuda for compilation')
    elif sys.argv[1] == 'mpi':
        print('Compiling mpi implementation:\n')
        compile_mpi()
    elif sys.argv[1] == 'cuda':
        print('Compiling cuda implementation:\n')
        compile_cuda()
    elif sys.argv[1] == 'cuda_clean':
        print('Removing .o files from cuda implementation:\n')
        clean('src_cuda/')
    elif sys.argv[1] == 'mpi_clean':
        print('Removing .o files from mpi implementation:\n')
        clean('src/')
    else:
        print('Error: specify either mpi or cuda for compilation')
    

if __name__ == '__main__':
    main()

