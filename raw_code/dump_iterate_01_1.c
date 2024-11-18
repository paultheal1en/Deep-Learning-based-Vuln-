static void dump_iterate(DumpState *s, Error **errp)

{

    GuestPhysBlock *block;

    int64_t size;

    int ret;

    Error *local_err = NULL;



    while (1) {

        block = s->next_block;



        size = block->target_end - block->target_start;

        if (s->has_filter) {

            size -= s->start;

            if (s->begin + s->length < block->target_end) {

                size -= block->target_end - (s->begin + s->length);

            }

        }

        write_memory(s, block, s->start, size, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }



        ret = get_next_block(s, block);

        if (ret == 1) {

            dump_completed(s);

        }

    }

}
