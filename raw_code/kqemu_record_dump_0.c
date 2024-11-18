void kqemu_record_dump(void)

{

    PCRecord **pr, *r;

    int i, h;

    FILE *f;

    int64_t total, sum;



    pr = malloc(sizeof(PCRecord *) * nb_pc_records);

    i = 0;

    total = 0;

    for(h = 0; h < PC_REC_HASH_SIZE; h++) {

        for(r = pc_rec_hash[h]; r != NULL; r = r->next) {

            pr[i++] = r;

            total += r->count;

        }

    }

    qsort(pr, nb_pc_records, sizeof(PCRecord *), pc_rec_cmp);



    f = fopen("/tmp/kqemu.stats", "w");

    if (!f) {

        perror("/tmp/kqemu.stats");

        exit(1);

    }

    fprintf(f, "total: %" PRId64 "\n", total);

    sum = 0;

    for(i = 0; i < nb_pc_records; i++) {

        r = pr[i];

        sum += r->count;

        fprintf(f, "%08lx: %" PRId64 " %0.2f%% %0.2f%%\n",

                r->pc,

                r->count,

                (double)r->count / (double)total * 100.0,

                (double)sum / (double)total * 100.0);

    }

    fclose(f);

    free(pr);



    kqemu_record_flush();

}
