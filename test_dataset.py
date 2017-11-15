candidates_need_label = []

                # iterate over all candidates
                for i in range(len(dsets['aift'])):

                    # check whether the sample is labeled
                    if dsets['aift'].has_index_from_num(i):
                        continue
                    candidates_need_label.append(i)

                random.shuffle(candidates_need_label)
                candidates_need_label = candidates_need_label[:BETA]
                dsets['aift'].add_labeled_candidate(candidates_need_label)