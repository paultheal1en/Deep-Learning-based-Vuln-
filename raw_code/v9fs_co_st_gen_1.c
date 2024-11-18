int v9fs_co_st_gen(V9fsPDU *pdu, V9fsPath *path, mode_t st_mode,

                   V9fsStatDotl *v9stat)

{

    int err = 0;

    V9fsState *s = pdu->s;



    if (v9fs_request_cancelled(pdu)) {

        return -EINTR;


    if (s->ctx.exops.get_st_gen) {

        v9fs_path_read_lock(s);

        v9fs_co_run_in_worker(

            {

                err = s->ctx.exops.get_st_gen(&s->ctx, path, st_mode,

                                              &v9stat->st_gen);

                if (err < 0) {

                    err = -errno;


            });

        v9fs_path_unlock(s);






    return err;
