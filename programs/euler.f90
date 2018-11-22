program euler
    use euler_diffusion
    implicit none

    real(dp) :: dx
    real(dp) :: T = 0.2d0

    real(dp), allocatable :: u(:,:)

    read(*,*) dx

    u = solve_euler(dx, T)

    block
        integer :: fu, i, Nx, Nt
        character(len=128) :: dxstr

        Nx = size(u, 1)
        Nt = size(u, 2)

        write(dxstr, "(es7.1)") dx
        open(newunit=fu, file="data/euler_" // trim(dxstr) // ".dat", status="replace")
        do i = 1, Nx
            write(fu, *) (i-1)*dx, u(i,1), u(i,Nt/2), u(i,Nt)
        end do
        close(fu)
    end block
end program
