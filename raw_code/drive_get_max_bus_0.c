int drive_get_max_bus(BlockInterfaceType type)

{

    int max_bus;

    DriveInfo *dinfo;



    max_bus = -1;

    TAILQ_FOREACH(dinfo, &drives, next) {

        if(dinfo->type == type &&

           dinfo->bus > max_bus)

            max_bus = dinfo->bus;

    }

    return max_bus;

}
