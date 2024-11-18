static bool ide_sect_range_ok(IDEState *s,

                              uint64_t sector, uint64_t nb_sectors)

{

    uint64_t total_sectors;



    bdrv_get_geometry(s->bs, &total_sectors);

    if (sector > total_sectors || nb_sectors > total_sectors - sector) {

        return false;

    }

    return true;

}
