DriveInfo *drive_get_by_id(const char *id)

{

    DriveInfo *dinfo;



    TAILQ_FOREACH(dinfo, &drives, next) {

        if (strcmp(id, dinfo->id))

            continue;

        return dinfo;

    }

    return NULL;

}
