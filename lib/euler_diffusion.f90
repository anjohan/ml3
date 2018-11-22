module euler_diffusion
    use iso_fortran_env, only: dp => real64
    implicit none
    real(dp), parameter :: pi = 4.0d0*atan(1.0d0)

    contains
        function solve_euler(dx, T) result(u)
            real(dp), intent(in) :: dx
            real(dp), intent(in), optional :: T
            integer :: Nx, Nt
            real(dp), allocatable :: u(:,:)

            integer :: i, j
            real(dp) :: dt, alpha

            Nx = nint(1.0d0/dx)
            dt = 0.5d0*dx**2
            alpha = dt/dx**2

            if (present(T)) then
                Nt = nint(T/dt)
            else
                Nt = nint(1.0d0/dt)
            end if

            write(*,*) Nx, Nt

            allocate(u(0:Nx,0:Nt))

            u(0,:) = 0
            u(Nx,:) = 0
            u(1:Nx-1,0) = [(sin(i*pi*dx), i = 1, Nx-1)]

            do j = 0, Nt-1
                !$omp parallel workshare
                u(1:Nx-1,j+1) = u(1:Nx-1,j) + alpha*(u(2:Nx,j) - 2*u(1:Nx-1,j) + u(0:Nx-2,j))
                !$omp end parallel workshare
            end do
        end function
end module
