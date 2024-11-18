static int check_bind(struct sockaddr *sa, socklen_t salen, bool *has_proto)

{

    int fd;



    fd = socket(sa->sa_family, SOCK_STREAM, 0);

    if (fd < 0) {

        return -1;

    }



    if (bind(fd, sa, salen) < 0) {

        close(fd);

        if (errno == EADDRNOTAVAIL) {

            *has_proto = false;

            return 0;

        }

        return -1;

    }



    close(fd);

    *has_proto = true;

    return 0;

}
