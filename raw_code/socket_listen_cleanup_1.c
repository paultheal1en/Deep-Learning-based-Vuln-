void socket_listen_cleanup(int fd, Error **errp)

{

    SocketAddress *addr;



    addr = socket_local_address(fd, errp);






    if (addr->type == SOCKET_ADDRESS_TYPE_UNIX

        && addr->u.q_unix.path) {

        if (unlink(addr->u.q_unix.path) < 0 && errno != ENOENT) {

            error_setg_errno(errp, errno,

                             "Failed to unlink socket %s",

                             addr->u.q_unix.path);





    qapi_free_SocketAddress(addr);
