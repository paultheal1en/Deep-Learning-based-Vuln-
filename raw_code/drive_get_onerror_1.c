BlockInterfaceErrorAction drive_get_onerror(BlockDriverState *bdrv)

{

    int index;



    for (index = 0; index < nb_drives; index++)

        if (drives_table[index].bdrv == bdrv)

            return drives_table[index].onerror;



    return BLOCK_ERR_REPORT;

}
