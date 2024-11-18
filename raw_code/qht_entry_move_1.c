qht_entry_move(struct qht_bucket *to, int i, struct qht_bucket *from, int j)

{

    qht_debug_assert(!(to == from && i == j));

    qht_debug_assert(to->pointers[i]);

    qht_debug_assert(from->pointers[j]);



    to->hashes[i] = from->hashes[j];

    atomic_set(&to->pointers[i], from->pointers[j]);



    from->hashes[j] = 0;

    atomic_set(&from->pointers[j], NULL);

}
