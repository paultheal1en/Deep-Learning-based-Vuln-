static int raw_fd_pool_get(BDRVRawState *s)

{

    int i;



    for (i = 0; i < RAW_FD_POOL_SIZE; i++) {

        /* already in use */

        if (s->fd_pool[i] != -1)

            continue;



        /* try to dup file descriptor */

        s->fd_pool[i] = dup(s->fd);

        if (s->fd_pool[i] != -1)

            return s->fd_pool[i];

    }



    /* we couldn't dup the file descriptor so just use the main one */

    return s->fd;

}
