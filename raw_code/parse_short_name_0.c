static int parse_short_name(BDRVVVFATState* s,

	long_file_name* lfn, direntry_t* direntry)

{

    int i, j;



    if (!is_short_name(direntry))

	return 1;



    for (j = 7; j >= 0 && direntry->name[j] == ' '; j--);

    for (i = 0; i <= j; i++) {

	if (direntry->name[i] <= ' ' || direntry->name[i] > 0x7f)

	    return -1;

	else if (s->downcase_short_names)

	    lfn->name[i] = qemu_tolower(direntry->name[i]);

	else

	    lfn->name[i] = direntry->name[i];

    }



    for (j = 2; j >= 0 && direntry->extension[j] == ' '; j--);

    if (j >= 0) {

	lfn->name[i++] = '.';

	lfn->name[i + j + 1] = '\0';

	for (;j >= 0; j--) {

	    if (direntry->extension[j] <= ' ' || direntry->extension[j] > 0x7f)

		return -2;

	    else if (s->downcase_short_names)

		lfn->name[i + j] = qemu_tolower(direntry->extension[j]);

	    else

		lfn->name[i + j] = direntry->extension[j];

	}

    } else

	lfn->name[i + j + 1] = '\0';



    lfn->len = strlen((char*)lfn->name);



    return 0;

}
