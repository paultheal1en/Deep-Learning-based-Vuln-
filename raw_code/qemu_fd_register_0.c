void qemu_fd_register(int fd)

{

    WSAEventSelect(fd, qemu_event_handle, FD_READ | FD_ACCEPT | FD_CLOSE |

                   FD_CONNECT | FD_WRITE | FD_OOB);

}
