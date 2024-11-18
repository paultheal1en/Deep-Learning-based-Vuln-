static inline void *host_from_stream_offset(QEMUFile *f,

                                            ram_addr_t offset,

                                            int flags)

{

    static RAMBlock *block = NULL;

    char id[256];

    uint8_t len;



    if (flags & RAM_SAVE_FLAG_CONTINUE) {

        if (!block || block->max_length <= offset) {

            error_report("Ack, bad migration stream!");

            return NULL;

        }



        return block->host + offset;

    }



    len = qemu_get_byte(f);

    qemu_get_buffer(f, (uint8_t *)id, len);

    id[len] = 0;



    QLIST_FOREACH_RCU(block, &ram_list.blocks, next) {

        if (!strncmp(id, block->idstr, sizeof(id)) &&

            block->max_length > offset) {

            return block->host + offset;

        }

    }



    error_report("Can't find block %s!", id);

    return NULL;

}
