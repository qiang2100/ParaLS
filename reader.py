import unicodedata


class Reader_lexical:
    def __init__(self):
        self.words_candidate = {}
        self.final_data = {}
        self.final_data_id = {}

    def create_feature(self, file_train):
        # side.n	303	11	if you want to find someone who can compose the biblical side , write us .
        with open(file_train, encoding='latin1') as fp:
            line = fp.readline()
            i = 0
            while line:
                context = line.split("\t")
                main_word = context[0]
                if main_word.split('.')[0] == "":
                    word = "."
                else:
                    word = main_word.split('.')[0]
                instance = context[1]
                word_index = context[2]
                sentence = self._clean_text(context[3].replace("\n", ""))
                if main_word not in self.words_candidate:
                    self.words_candidate[main_word] = {}
                if instance not in self.words_candidate[main_word]:
                    self.words_candidate[main_word][instance] = []
                self.words_candidate[main_word][instance].append([word, sentence, word_index])
                line = fp.readline()
        return

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def create_candidates(self, file_path_candidate):
        with open(file_path_candidate, encoding='latin1') as fp:
            line = fp.readline()
            self.candidates = {}
            while line:
                word = line.split("::")[0]
                # if word == "..N":
                self.candidates[word] = []
                candidates_words = line.split("::")[1]
                for candidate_word in candidates_words.split(";"):
                    if ((len(candidate_word.split(' ')) > 1) or (len(candidate_word.split('-')) > 1)) or len(
                            candidate_word) < 1:
                        pass
                    else:
                        self.candidates[word].append(candidate_word.replace("\n", ""))
                line = fp.readline()
        return

    def read_eval_line(self, eval_line, ignore_mwe=True):
        eval_weights = {}
        segments = eval_line.split("\t")
        instance_id = segments[1].strip()
        for candidate_weight in segments[2:]:
            if len(candidate_weight) > 0:
                delimiter_ind = candidate_weight.rfind(' ')
                candidate = candidate_weight[:delimiter_ind]
                weight = candidate_weight[delimiter_ind:]
                if ignore_mwe and ((len(candidate.split(' ')) > 1) or (len(candidate.split('-')) > 1)):
                    continue
                try:
                    eval_weights[candidate] = float(weight)
                    # eval_weights.append((candidate, float(weight)))
                except:
                    print("Error appending: %s %s" % (candidate, weight))

        return instance_id, eval_weights

    # def created_dict_proposed(self, proposed_words_gap, proposed_words_scores):
    #     proposed_words = {}
    #     for i in range(0, len(proposed_words_gap)):
    #         proposed_words[proposed_words_gap[i]] = proposed_words_scores[i]

    #     return proposed_words
    def created_dict_proposed(self, proposed_words_gap):
        proposed_words = {}

        for i in range(0, len(proposed_words_gap)):
            proposed_words[proposed_words_gap[i]] = 0

        return proposed_words