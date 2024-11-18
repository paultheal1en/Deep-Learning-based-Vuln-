static void secondary_do_checkpoint(BDRVReplicationState *s, Error **errp)
{
    Error *local_err = NULL;
    int ret;
    if (!s->secondary_disk->bs->job) {
        error_setg(errp, "Backup job was cancelled unexpectedly");
    backup_do_checkpoint(s->secondary_disk->bs->job, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
    ret = s->active_disk->bs->drv->bdrv_make_empty(s->active_disk->bs);
    if (ret < 0) {
        error_setg(errp, "Cannot make active disk empty");
    ret = s->hidden_disk->bs->drv->bdrv_make_empty(s->hidden_disk->bs);
    if (ret < 0) {
        error_setg(errp, "Cannot make hidden disk empty");