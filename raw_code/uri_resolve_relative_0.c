uri_resolve_relative (const char *uri, const char * base)

{

    char *val = NULL;

    int ret;

    int ix;

    int pos = 0;

    int nbslash = 0;

    int len;

    URI *ref = NULL;

    URI *bas = NULL;

    char *bptr, *uptr, *vptr;

    int remove_path = 0;



    if ((uri == NULL) || (*uri == 0))

	return NULL;



    /*

     * First parse URI into a standard form

     */

    ref = uri_new ();

    /* If URI not already in "relative" form */

    if (uri[0] != '.') {

	ret = uri_parse_into (ref, uri);

	if (ret != 0)

	    goto done;		/* Error in URI, return NULL */

    } else

	ref->path = g_strdup(uri);



    /*

     * Next parse base into the same standard form

     */

    if ((base == NULL) || (*base == 0)) {

	val = g_strdup (uri);

	goto done;

    }

    bas = uri_new ();

    if (base[0] != '.') {

	ret = uri_parse_into (bas, base);

	if (ret != 0)

	    goto done;		/* Error in base, return NULL */

    } else

	bas->path = g_strdup(base);



    /*

     * If the scheme / server on the URI differs from the base,

     * just return the URI

     */

    if ((ref->scheme != NULL) &&

	((bas->scheme == NULL) ||

	 (strcmp (bas->scheme, ref->scheme)) ||

	 (strcmp (bas->server, ref->server)))) {

	val = g_strdup (uri);

	goto done;

    }

    if (!strcmp(bas->path, ref->path)) {

	val = g_strdup("");

	goto done;

    }

    if (bas->path == NULL) {

	val = g_strdup(ref->path);

	goto done;

    }

    if (ref->path == NULL) {

        ref->path = (char *) "/";

	remove_path = 1;

    }



    /*

     * At this point (at last!) we can compare the two paths

     *

     * First we take care of the special case where either of the

     * two path components may be missing (bug 316224)

     */

    if (bas->path == NULL) {

	if (ref->path != NULL) {

	    uptr = ref->path;

	    if (*uptr == '/')

		uptr++;

	    /* exception characters from uri_to_string */

	    val = uri_string_escape(uptr, "/;&=+$,");

	}

	goto done;

    }

    bptr = bas->path;

    if (ref->path == NULL) {

	for (ix = 0; bptr[ix] != 0; ix++) {

	    if (bptr[ix] == '/')

		nbslash++;

	}

	uptr = NULL;

	len = 1;	/* this is for a string terminator only */

    } else {

    /*

     * Next we compare the two strings and find where they first differ

     */

	if ((ref->path[pos] == '.') && (ref->path[pos+1] == '/'))

            pos += 2;

	if ((*bptr == '.') && (bptr[1] == '/'))

            bptr += 2;

	else if ((*bptr == '/') && (ref->path[pos] != '/'))

	    bptr++;

	while ((bptr[pos] == ref->path[pos]) && (bptr[pos] != 0))

	    pos++;



	if (bptr[pos] == ref->path[pos]) {

	    val = g_strdup("");

	    goto done;		/* (I can't imagine why anyone would do this) */

	}



	/*

	 * In URI, "back up" to the last '/' encountered.  This will be the

	 * beginning of the "unique" suffix of URI

	 */

	ix = pos;

	if ((ref->path[ix] == '/') && (ix > 0))

	    ix--;

	else if ((ref->path[ix] == 0) && (ix > 1) && (ref->path[ix - 1] == '/'))

	    ix -= 2;

	for (; ix > 0; ix--) {

	    if (ref->path[ix] == '/')

		break;

	}

	if (ix == 0) {

	    uptr = ref->path;

	} else {

	    ix++;

	    uptr = &ref->path[ix];

	}



	/*

	 * In base, count the number of '/' from the differing point

	 */

	if (bptr[pos] != ref->path[pos]) {/* check for trivial URI == base */

	    for (; bptr[ix] != 0; ix++) {

		if (bptr[ix] == '/')

		    nbslash++;

	    }

	}

	len = strlen (uptr) + 1;

    }



    if (nbslash == 0) {

	if (uptr != NULL)

	    /* exception characters from uri_to_string */

	    val = uri_string_escape(uptr, "/;&=+$,");

	goto done;

    }



    /*

     * Allocate just enough space for the returned string -

     * length of the remainder of the URI, plus enough space

     * for the "../" groups, plus one for the terminator

     */

    val = g_malloc (len + 3 * nbslash);

    vptr = val;

    /*

     * Put in as many "../" as needed

     */

    for (; nbslash>0; nbslash--) {

	*vptr++ = '.';

	*vptr++ = '.';

	*vptr++ = '/';

    }

    /*

     * Finish up with the end of the URI

     */

    if (uptr != NULL) {

        if ((vptr > val) && (len > 0) &&

	    (uptr[0] == '/') && (vptr[-1] == '/')) {

	    memcpy (vptr, uptr + 1, len - 1);

	    vptr[len - 2] = 0;

	} else {

	    memcpy (vptr, uptr, len);

	    vptr[len - 1] = 0;

	}

    } else {

	vptr[len - 1] = 0;

    }



    /* escape the freshly-built path */

    vptr = val;

	/* exception characters from uri_to_string */

    val = uri_string_escape(vptr, "/;&=+$,");

    g_free(vptr);



done:

    /*

     * Free the working variables

     */

    if (remove_path != 0)

        ref->path = NULL;

    if (ref != NULL)

	uri_free (ref);

    if (bas != NULL)

	uri_free (bas);



    return val;

}
