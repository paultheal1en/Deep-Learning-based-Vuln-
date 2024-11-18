void *fw_cfg_modify_file(FWCfgState *s, const char *filename,

                        void *data, size_t len)

{

    int i, index;

    void *ptr = NULL;



    assert(s->files);



    index = be32_to_cpu(s->files->count);

    assert(index < fw_cfg_file_slots(s));



    for (i = 0; i < index; i++) {

        if (strcmp(filename, s->files->f[i].name) == 0) {

            ptr = fw_cfg_modify_bytes_read(s, FW_CFG_FILE_FIRST + i,

                                           data, len);

            s->files->f[i].size   = cpu_to_be32(len);

            return ptr;

        }

    }

    /* add new one */

    fw_cfg_add_file_callback(s, filename, NULL, NULL, NULL, data, len, true);

    return NULL;

}
