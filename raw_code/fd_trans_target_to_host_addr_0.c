static TargetFdAddrFunc fd_trans_target_to_host_addr(int fd)

{

    if (fd < target_fd_max && target_fd_trans[fd]) {

        return target_fd_trans[fd]->target_to_host_addr;

    }

    return NULL;

}
