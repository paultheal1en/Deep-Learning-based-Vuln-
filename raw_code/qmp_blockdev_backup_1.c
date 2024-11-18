void qmp_blockdev_backup(BlockdevBackup *arg, Error **errp)

{

    do_blockdev_backup(arg, NULL, errp);

}
