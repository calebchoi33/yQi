from tagging.parse_books import parse_book
from collections import deque

DESIRED_TOKENS = 1000

def get_relevant_sections(book_path: str, chapter_idx: int, section_idx: int) -> str:
    """
    Gets the text of the relevant sections from the book.
    It returns the text of the section at the given chapter and section indices, plus the text of sections surrounding it, based on the desired number of total tokens.
    It also respects chapter boundaries and will prefer to return sections from the same chapter

    Args:
        book_path: str
            The path to the book file.
        chapter_idx: int
            The index of the chapter.
        section_idx: int
            The index of the section.
    """

    parsed_book = parse_book(book_path)

    curr_sections = parsed_book[chapter_idx]["sections"]
    curr_section_text = curr_sections[section_idx]["section_title"] + curr_sections[section_idx]["section_text"]
    sections_result = deque([curr_section_text,])
    total_tokens = len(curr_section_text)
    prev_section_idx = section_idx - 1
    next_section_idx = section_idx + 1

    while (prev_section_idx >= 0 or next_section_idx < len(curr_sections)) and total_tokens < DESIRED_TOKENS:
        print('before prev_section', total_tokens)
        # take the previous section
        if prev_section_idx >= 0:
            curr_section_text = curr_sections[prev_section_idx]["section_title"] + curr_sections[prev_section_idx]["section_text"]
            tokens_remaining = DESIRED_TOKENS-total_tokens
            # only take the remaining tokens
            if len(curr_section_text) > tokens_remaining:
                curr_section_text = curr_section_text[len(curr_section_text)-tokens_remaining:]
            
            sections_result.appendleft(curr_section_text)
            total_tokens += len(curr_section_text)
            prev_section_idx -= 1
        
        if total_tokens >= DESIRED_TOKENS:
            break
        
        print('before next_section', total_tokens)
        # take the next section
        if next_section_idx < len(curr_sections):
            curr_section_text = curr_sections[next_section_idx]["section_title"] + curr_sections[next_section_idx]["section_text"]
            tokens_remaining = DESIRED_TOKENS-total_tokens
            if len(curr_section_text) > tokens_remaining:
                curr_section_text = curr_section_text[:tokens_remaining]
            sections_result.append(curr_section_text)
            total_tokens += len(curr_section_text)
            next_section_idx += 1
        

    print('final', total_tokens)
    print("--------------------------------")
    return "\n\n".join(sections_result)


print(get_relevant_sections("books/《人紀傷寒論》.txt", 4, 44))