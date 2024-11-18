void blk_resume_after_migration(Error **errp)

{

    BlockBackend *blk;

    Error *local_err = NULL;



    for (blk = blk_all_next(NULL); blk; blk = blk_all_next(blk)) {

        if (!blk->disable_perm) {

            continue;

        }



        blk->disable_perm = false;



        blk_set_perm(blk, blk->perm, blk->shared_perm, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            blk->disable_perm = true;

            return;

        }

    }

}
