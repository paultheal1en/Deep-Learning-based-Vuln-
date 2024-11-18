static void add_index_entry(AVStream *st,

                            int64_t pos, int64_t timestamp, int flags)

{

    AVIndexEntry *entries, *ie;

    

    entries = av_fast_realloc(st->index_entries,

                              &st->index_entries_allocated_size,

                              (st->nb_index_entries + 1) * 

                              sizeof(AVIndexEntry));

    if (entries) {

        st->index_entries = entries;

        ie = &entries[st->nb_index_entries++];

        ie->pos = pos;

        ie->timestamp = timestamp;

        ie->flags = flags;

    }

}
