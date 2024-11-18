static void cleanup_infolist(CommandLineParameterInfoList *head)

{

    CommandLineParameterInfoList *pre_entry, *cur, *del_entry;



    cur = head;

    while (cur->next) {

        pre_entry = head;

        while (pre_entry != cur->next) {

            if (!strcmp(pre_entry->value->name, cur->next->value->name)) {

                del_entry = cur->next;

                cur->next = cur->next->next;

                g_free(del_entry);

                break;

            }

            pre_entry = pre_entry->next;

        }

        cur = cur->next;

    }

}
