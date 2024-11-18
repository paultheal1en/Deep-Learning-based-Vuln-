FsTypeEntry *get_fsdev_fsentry(char *id)

{

    struct FsTypeListEntry *fsle;



    QTAILQ_FOREACH(fsle, &fstype_entries, next) {

        if (strcmp(fsle->fse.fsdev_id, id) == 0) {

            return &fsle->fse;

        }

    }

    return NULL;

}
