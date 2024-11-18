static void clear_commits(BDRVVVFATState* s)

{

    int i;

DLOG(fprintf(stderr, "clear_commits (%d commits)\n", s->commits.next));

    for (i = 0; i < s->commits.next; i++) {

	commit_t* commit = array_get(&(s->commits), i);

	assert(commit->path || commit->action == ACTION_WRITEOUT);

	if (commit->action != ACTION_WRITEOUT) {

	    assert(commit->path);

	    free(commit->path);

	} else

	    assert(commit->path == NULL);

    }

    s->commits.next = 0;

}
