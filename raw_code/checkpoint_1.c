static void checkpoint(void) {

    assert(((mapping_t*)array_get(&(vvv->mapping), 0))->end == 2);

    check1(vvv);

    check2(vvv);

    assert(!vvv->current_mapping || vvv->current_fd || (vvv->current_mapping->mode & MODE_DIRECTORY));

#if 0

    if (((direntry_t*)vvv->directory.pointer)[1].attributes != 0xf)

	fprintf(stderr, "Nonono!\n");

    mapping_t* mapping;

    direntry_t* direntry;

    assert(vvv->mapping.size >= vvv->mapping.item_size * vvv->mapping.next);

    assert(vvv->directory.size >= vvv->directory.item_size * vvv->directory.next);

    if (vvv->mapping.next<47)

	return;

    assert((mapping = array_get(&(vvv->mapping), 47)));

    assert(mapping->dir_index < vvv->directory.next);

    direntry = array_get(&(vvv->directory), mapping->dir_index);

    assert(!memcmp(direntry->name, "USB     H  ", 11) || direntry->name[0]==0);

#endif

    return;

    /* avoid compiler warnings: */

    hexdump(NULL, 100);

    remove_mapping(vvv, NULL);

    print_mapping(NULL);

    print_direntry(NULL);

}
