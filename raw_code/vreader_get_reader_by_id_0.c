vreader_get_reader_by_id(vreader_id_t id)

{

    VReader *reader = NULL;

    VReaderListEntry *current_entry = NULL;



    if (id == (vreader_id_t) -1) {

        return NULL;

    }



    vreader_list_lock();

    for (current_entry = vreader_list_get_first(vreader_list); current_entry;

            current_entry = vreader_list_get_next(current_entry)) {

        VReader *creader = vreader_list_get_reader(current_entry);

        if (creader->id == id) {

            reader = creader;

            break;

        }

        vreader_free(creader);

    }

    vreader_list_unlock();

    return reader;

}
