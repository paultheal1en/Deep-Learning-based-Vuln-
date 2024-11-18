static int qcow2_truncate(BlockDriverState *bs, int64_t offset)

{

    BDRVQcowState *s = bs->opaque;

    int ret, new_l1_size;



    if (offset & 511) {

        return -EINVAL;

    }



    /* cannot proceed if image has snapshots */

    if (s->nb_snapshots) {

        return -ENOTSUP;

    }



    /* shrinking is currently not supported */

    if (offset < bs->total_sectors * 512) {

        return -ENOTSUP;

    }



    new_l1_size = size_to_l1(s, offset);

    ret = qcow2_grow_l1_table(bs, new_l1_size);

    if (ret < 0) {

        return ret;

    }



    /* write updated header.size */

    offset = cpu_to_be64(offset);

    ret = bdrv_pwrite(bs->file, offsetof(QCowHeader, size),

                      &offset, sizeof(uint64_t));

    if (ret < 0) {

        return ret;

    }



    s->l1_vm_state_index = new_l1_size;

    return 0;

}
