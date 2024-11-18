vreader_copy_list(VReaderList *list)

{

    VReaderList *new_list = NULL;

    VReaderListEntry *current_entry = NULL;



    new_list = vreader_list_new();

    if (new_list == NULL) {

        return NULL;

    }

    for (current_entry = vreader_list_get_first(list); current_entry;

         current_entry = vreader_list_get_next(current_entry)) {

        VReader *reader = vreader_list_get_reader(current_entry);

        VReaderListEntry *new_entry = vreader_list_entry_new(reader);



        vreader_free(reader);

        vreader_queue(new_list, new_entry);

    }

    return new_list;

}
