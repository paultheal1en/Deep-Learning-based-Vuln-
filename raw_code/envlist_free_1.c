envlist_free(envlist_t *envlist)

{

	struct envlist_entry *entry;



	assert(envlist != NULL);



	while (envlist->el_entries.lh_first != NULL) {

		entry = envlist->el_entries.lh_first;

		QLIST_REMOVE(entry, ev_link);



		free((char *)entry->ev_var);

		free(entry);

	}

	free(envlist);

}
