static inline int array_roll(array_t* array,int index_to,int index_from,int count)

{

    char* buf;

    char* from;

    char* to;

    int is;



    if(!array ||

	    index_to<0 || index_to>=array->next ||

	    index_from<0 || index_from>=array->next)

	return -1;



    if(index_to==index_from)

	return 0;



    is=array->item_size;

    from=array->pointer+index_from*is;

    to=array->pointer+index_to*is;

    buf=g_malloc(is*count);

    memcpy(buf,from,is*count);



    if(index_to<index_from)

	memmove(to+is*count,to,from-to);

    else

	memmove(from,from+is*count,to-from);



    memcpy(to,buf,is*count);



    free(buf);



    return 0;

}
