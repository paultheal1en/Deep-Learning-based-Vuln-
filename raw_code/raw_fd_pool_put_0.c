static void raw_fd_pool_put(RawAIOCB *acb)

{

    BDRVRawState *s = acb->common.bs->opaque;

    int i;



    for (i = 0; i < RAW_FD_POOL_SIZE; i++) {

        if (s->fd_pool[i] == acb->fd) {

            close(s->fd_pool[i]);

            s->fd_pool[i] = -1;

        }

    }

}
