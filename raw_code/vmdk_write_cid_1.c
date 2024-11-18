static int vmdk_write_cid(BlockDriverState *bs, uint32_t cid)

{

    char desc[DESC_SIZE], tmp_desc[DESC_SIZE];

    char *p_name, *tmp_str;



    /* the descriptor offset = 0x200 */

    if (bdrv_pread(bs->file, 0x200, desc, DESC_SIZE) != DESC_SIZE)

        return -1;



    tmp_str = strstr(desc,"parentCID");

    pstrcpy(tmp_desc, sizeof(tmp_desc), tmp_str);

    if ((p_name = strstr(desc,"CID")) != NULL) {

        p_name += sizeof("CID");

        snprintf(p_name, sizeof(desc) - (p_name - desc), "%x\n", cid);

        pstrcat(desc, sizeof(desc), tmp_desc);

    }



    if (bdrv_pwrite(bs->file, 0x200, desc, DESC_SIZE) != DESC_SIZE)

        return -1;

    return 0;

}
