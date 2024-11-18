static void kqemu_record_flush(void)

{

    PCRecord *r, *r_next;

    int h;



    for(h = 0; h < PC_REC_HASH_SIZE; h++) {

        for(r = pc_rec_hash[h]; r != NULL; r = r_next) {

            r_next = r->next;

            free(r);

        }

        pc_rec_hash[h] = NULL;

    }

    nb_pc_records = 0;

}
