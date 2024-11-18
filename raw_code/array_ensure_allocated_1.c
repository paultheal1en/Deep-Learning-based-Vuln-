static inline int array_ensure_allocated(array_t* array, int index)

{

    if((index + 1) * array->item_size > array->size) {

        int new_size = (index + 32) * array->item_size;

        array->pointer = g_realloc(array->pointer, new_size);

        if (!array->pointer)

            return -1;


        array->size = new_size;

        array->next = index + 1;

    }



    return 0;

}