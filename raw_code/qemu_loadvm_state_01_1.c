static int qemu_loadvm_state(QEMUFile *f)

{

    SaveStateEntry *se;

    int len, ret, instance_id, record_len, version_id;

    int64_t total_len, end_pos, cur_pos;

    unsigned int v;

    char idstr[256];



    v = qemu_get_be32(f);

    if (v != QEMU_VM_FILE_MAGIC)

        goto fail;

    v = qemu_get_be32(f);

    if (v != QEMU_VM_FILE_VERSION) {

    fail:

        ret = -1;

        goto the_end;

    }

    total_len = qemu_get_be64(f);

    end_pos = total_len + qemu_ftell(f);

    for(;;) {

        if (qemu_ftell(f) >= end_pos)

            break;

        len = qemu_get_byte(f);

        qemu_get_buffer(f, (uint8_t *)idstr, len);

        idstr[len] = '\0';

        instance_id = qemu_get_be32(f);

        version_id = qemu_get_be32(f);

        record_len = qemu_get_be32(f);

#if 0

        printf("idstr=%s instance=0x%x version=%d len=%d\n",

               idstr, instance_id, version_id, record_len);

#endif

        cur_pos = qemu_ftell(f);

        se = find_se(idstr, instance_id);

        if (!se) {

            fprintf(stderr, "qemu: warning: instance 0x%x of device '%s' not present in current VM\n",

                    instance_id, idstr);

        } else {

            ret = se->load_state(f, se->opaque, version_id);

            if (ret < 0) {

                fprintf(stderr, "qemu: warning: error while loading state for instance 0x%x of device '%s'\n",

                        instance_id, idstr);

            }

        }

        /* always seek to exact end of record */

        qemu_fseek(f, cur_pos + record_len, SEEK_SET);

    }

    ret = 0;

 the_end:

    return ret;

}
