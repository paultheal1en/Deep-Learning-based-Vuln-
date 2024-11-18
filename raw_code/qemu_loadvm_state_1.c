int qemu_loadvm_state(QEMUFile *f)

{

    QLIST_HEAD(, LoadStateEntry) loadvm_handlers =

        QLIST_HEAD_INITIALIZER(loadvm_handlers);

    LoadStateEntry *le, *new_le;

    Error *local_err = NULL;

    uint8_t section_type;

    unsigned int v;

    int ret;

    int file_error_after_eof = -1;



    if (qemu_savevm_state_blocked(&local_err)) {

        error_report_err(local_err);

        return -EINVAL;

    }



    v = qemu_get_be32(f);

    if (v != QEMU_VM_FILE_MAGIC) {

        error_report("Not a migration stream");

        return -EINVAL;

    }



    v = qemu_get_be32(f);

    if (v == QEMU_VM_FILE_VERSION_COMPAT) {

        error_report("SaveVM v2 format is obsolete and don't work anymore");

        return -ENOTSUP;

    }

    if (v != QEMU_VM_FILE_VERSION) {

        error_report("Unsupported migration stream version");

        return -ENOTSUP;

    }



    while ((section_type = qemu_get_byte(f)) != QEMU_VM_EOF) {

        uint32_t instance_id, version_id, section_id;

        SaveStateEntry *se;

        char idstr[257];

        int len;



        trace_qemu_loadvm_state_section(section_type);

        switch (section_type) {

        case QEMU_VM_SECTION_START:

        case QEMU_VM_SECTION_FULL:

            /* Read section start */

            section_id = qemu_get_be32(f);

            len = qemu_get_byte(f);

            qemu_get_buffer(f, (uint8_t *)idstr, len);

            idstr[len] = 0;

            instance_id = qemu_get_be32(f);

            version_id = qemu_get_be32(f);



            trace_qemu_loadvm_state_section_startfull(section_id, idstr,

                                                      instance_id, version_id);

            /* Find savevm section */

            se = find_se(idstr, instance_id);

            if (se == NULL) {

                error_report("Unknown savevm section or instance '%s' %d",

                             idstr, instance_id);

                ret = -EINVAL;

                goto out;

            }



            /* Validate version */

            if (version_id > se->version_id) {

                error_report("savevm: unsupported version %d for '%s' v%d",

                             version_id, idstr, se->version_id);

                ret = -EINVAL;

                goto out;

            }



            /* Add entry */

            le = g_malloc0(sizeof(*le));



            le->se = se;

            le->section_id = section_id;

            le->version_id = version_id;

            QLIST_INSERT_HEAD(&loadvm_handlers, le, entry);



            ret = vmstate_load(f, le->se, le->version_id);

            if (ret < 0) {

                error_report("error while loading state for instance 0x%x of"

                             " device '%s'", instance_id, idstr);

                goto out;

            }

            break;

        case QEMU_VM_SECTION_PART:

        case QEMU_VM_SECTION_END:

            section_id = qemu_get_be32(f);



            trace_qemu_loadvm_state_section_partend(section_id);

            QLIST_FOREACH(le, &loadvm_handlers, entry) {

                if (le->section_id == section_id) {

                    break;

                }

            }

            if (le == NULL) {

                error_report("Unknown savevm section %d", section_id);

                ret = -EINVAL;

                goto out;

            }



            ret = vmstate_load(f, le->se, le->version_id);

            if (ret < 0) {

                error_report("error while loading state section id %d(%s)",

                             section_id, le->se->idstr);

                goto out;

            }

            break;

        default:

            error_report("Unknown savevm section type %d", section_type);

            ret = -EINVAL;

            goto out;

        }

    }



    file_error_after_eof = qemu_file_get_error(f);



    /*

     * Try to read in the VMDESC section as well, so that dumping tools that

     * intercept our migration stream have the chance to see it.

     */

    if (qemu_get_byte(f) == QEMU_VM_VMDESCRIPTION) {

        uint32_t size = qemu_get_be32(f);

        uint8_t *buf = g_malloc(0x1000);



        while (size > 0) {

            uint32_t read_chunk = MIN(size, 0x1000);

            qemu_get_buffer(f, buf, read_chunk);

            size -= read_chunk;

        }

        g_free(buf);

    }



    cpu_synchronize_all_post_init();



    ret = 0;



out:

    QLIST_FOREACH_SAFE(le, &loadvm_handlers, entry, new_le) {

        QLIST_REMOVE(le, entry);

        g_free(le);

    }



    if (ret == 0) {

        /* We may not have a VMDESC section, so ignore relative errors */

        ret = file_error_after_eof;

    }



    return ret;

}
