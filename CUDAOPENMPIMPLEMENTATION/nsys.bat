@echo off

set nsight-systems="C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe"

IF "%1"=="" (
    echo "Usage: %0 <executable> <run args>"
    pause
    goto :eof
)

%nsight-systems% profile --trace=cuda --sample=none -o nsys_report %*