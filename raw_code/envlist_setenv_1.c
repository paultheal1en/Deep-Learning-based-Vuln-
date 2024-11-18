envlist_setenv(envlist_t *envlist, const char *env)

{

	struct envlist_entry *entry = NULL;

	const char *eq_sign;

	size_t envname_len;



	if ((envlist == NULL) || (env == NULL))

		return (EINVAL);



	/* find out first equals sign in given env */

	if ((eq_sign = strchr(env, '=')) == NULL)

		return (EINVAL);

	envname_len = eq_sign - env + 1;



	/*

	 * If there already exists variable with given name

	 * we remove and release it before allocating a whole

	 * new entry.

	 */

	for (entry = envlist->el_entries.lh_first; entry != NULL;

	    entry = entry->ev_link.le_next) {

		if (strncmp(entry->ev_var, env, envname_len) == 0)

			break;

	}



	if (entry != NULL) {

		QLIST_REMOVE(entry, ev_link);

		free((char *)entry->ev_var);

		free(entry);

	} else {

		envlist->el_count++;

	}



	if ((entry = malloc(sizeof (*entry))) == NULL)

		return (errno);

	if ((entry->ev_var = strdup(env)) == NULL) {

		free(entry);

		return (errno);

	}

	QLIST_INSERT_HEAD(&envlist->el_entries, entry, ev_link);



	return (0);

}
