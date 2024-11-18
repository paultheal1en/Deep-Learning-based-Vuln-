static int qcow_write_snapshots(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    QCowSnapshot *sn;

    QCowSnapshotHeader h;

    int i, name_size, id_str_size, snapshots_size;

    uint64_t data64;

    uint32_t data32;

    int64_t offset, snapshots_offset;



    /* compute the size of the snapshots */

    offset = 0;

    for(i = 0; i < s->nb_snapshots; i++) {

        sn = s->snapshots + i;

        offset = align_offset(offset, 8);

        offset += sizeof(h);

        offset += strlen(sn->id_str);

        offset += strlen(sn->name);

    }

    snapshots_size = offset;



    snapshots_offset = qcow2_alloc_clusters(bs, snapshots_size);

    offset = snapshots_offset;

    if (offset < 0) {

        return offset;

    }



    for(i = 0; i < s->nb_snapshots; i++) {

        sn = s->snapshots + i;

        memset(&h, 0, sizeof(h));

        h.l1_table_offset = cpu_to_be64(sn->l1_table_offset);

        h.l1_size = cpu_to_be32(sn->l1_size);

        h.vm_state_size = cpu_to_be32(sn->vm_state_size);

        h.date_sec = cpu_to_be32(sn->date_sec);

        h.date_nsec = cpu_to_be32(sn->date_nsec);

        h.vm_clock_nsec = cpu_to_be64(sn->vm_clock_nsec);



        id_str_size = strlen(sn->id_str);

        name_size = strlen(sn->name);

        h.id_str_size = cpu_to_be16(id_str_size);

        h.name_size = cpu_to_be16(name_size);

        offset = align_offset(offset, 8);

        if (bdrv_pwrite(bs->file, offset, &h, sizeof(h)) != sizeof(h))

            goto fail;

        offset += sizeof(h);

        if (bdrv_pwrite(bs->file, offset, sn->id_str, id_str_size) != id_str_size)

            goto fail;

        offset += id_str_size;

        if (bdrv_pwrite(bs->file, offset, sn->name, name_size) != name_size)

            goto fail;

        offset += name_size;

    }



    /* update the various header fields */

    data64 = cpu_to_be64(snapshots_offset);

    if (bdrv_pwrite(bs->file, offsetof(QCowHeader, snapshots_offset),

                    &data64, sizeof(data64)) != sizeof(data64))

        goto fail;

    data32 = cpu_to_be32(s->nb_snapshots);

    if (bdrv_pwrite(bs->file, offsetof(QCowHeader, nb_snapshots),

                    &data32, sizeof(data32)) != sizeof(data32))

        goto fail;



    /* free the old snapshot table */

    qcow2_free_clusters(bs, s->snapshots_offset, s->snapshots_size);

    s->snapshots_offset = snapshots_offset;

    s->snapshots_size = snapshots_size;

    return 0;

 fail:

    return -1;

}
