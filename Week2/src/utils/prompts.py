import os
from typing import Literal

import numpy as np


class Prompt:
    """
    Class for managing prompts.
    """

    def __init__(self, mode: Literal["text", "point", "bbox"]):
        """Mode determines the modality of this prompting family (text, points, bboxes)."""
        self.mode = mode
        self.prompts = []

    def get_all_prompts(self):
        return self.prompts

    def clean_history(self):
        """Delete the entire prompts history."""
        self.prompts = []

    def add_prompt(self, prompt):
        """Add new prompt to history."""
        if isinstance(prompt, str) and self.mode == "text":
            self.prompts.append(prompt)
            return

        if (
            isinstance(prompt, np.ndarray)
            and self.mode == "point"
            and prompt.shape[-1] == 2
        ):
            self.prompts.append(prompt)
            return

        if (
            isinstance(prompt, np.ndarray)
            and self.mode == "bbox"
            and prompt.shape[-1] == 4
        ):
            self.prompts.append(prompt)
            return

        else:
            print(
                f"WARNING: Mismatch between prompt history mode '{self.mode}' and added prompt. Abort adding prompt."
            )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    @classmethod
    def from_file(cls, file: str):
        """Load previously stored prompts from a .txt file."""
        if not os.path.exists(file):
            print(f"WARNING: {file} does not exist.")
            return

        extension = os.path.splitext(file)[1]
        prompts = []
        if extension == ".npy":
            with open(file, "rb") as f:
                prompts.append(np.load(f))
            if prompts[0].shape[1] == 4:
                mode = "bbox"
            else:
                mode = "point"
        else:
            with open(file, "r") as f:
                mode = "text"
                for line in f.readlines():
                    prompts.append(line.strip())
        instance = cls(mode)
        instance.prompts = prompts
        return instance

    def save_prompt_history(self, file: str, force: bool = False):
        """Save history of prompts to a .txt or .npy file, depending on the prompts mode."""
        if len(self.prompts) == 0:
            print("WARNING: Prompt history is empty. Saving aborted.")
            return

        exists = os.path.exists(file)
        if exists:
            if not force:
                print(
                    f"WARNING: {file} already exists. Please, change the file name to avoid overwriting other files."
                )
                return
            print(f"WARNING: Overwriting {file}.")

        if self.mode == "text":
            file = os.path.splitext(file)[0] + ".txt"
            with open(file, "w") as f:
                for prompt in self.prompts:
                    line = f"{prompt}\n"
                    f.write(line)
        else:
            file = os.path.splitext(file)[0] + ".npy"

            with open(file, "wb") as f:
                for prompt in self.prompts:
                    np.save(f, np.array(prompt))


if __name__ == "__main__":
    from pprint import pprint

    def check_contents(file: str, expected):
        print("\nFile contents:")
        with open(file, "r") as f:
            for line in f.readlines():
                print(line)
        print("\nExpected contents:")
        pprint(expected)

    prompts_folder = "prompts/test"
    text_file = "text_test.txt"
    point_file = "point_test.npy"
    bbox_file = "bbox_test.npy"
    os.makedirs(prompts_folder, exist_ok=True)

    # Deleting preexisting test files
    if os.path.exists(os.path.join(prompts_folder, text_file)):
        os.remove(os.path.join(prompts_folder, text_file))
    if os.path.exists(os.path.join(prompts_folder, point_file)):
        os.remove(os.path.join(prompts_folder, point_file))
    if os.path.exists(os.path.join(prompts_folder, bbox_file)):
        os.remove(os.path.join(prompts_folder, bbox_file))

    # Test saving files
    print("===== Test Saving Prompts =====")

    ## Text prompts
    print("\nText saving.")
    p = Prompt("text")
    p1 = "Awelo de piña"
    p2 = "Caramelo pato."
    file = os.path.join(prompts_folder, text_file)
    p.add_prompt(p1)
    p.add_prompt(p2)
    p.save_prompt_history(file)

    ## Points prompts
    print("\nPoints saving.")
    p = Prompt("point")
    p1 = np.random.randint(0, 255, [10, 2])
    p2 = np.random.randint(0, 255, [78, 2])
    file = os.path.join(prompts_folder, point_file)

    p.add_prompt(p1)
    p.add_prompt(p2)
    p.save_prompt_history(file)

    ## Bbox prompts
    print("\nBbox saving.")
    p = Prompt("bbox")
    p1 = np.random.randint(0, 255, [65, 4])
    p2 = np.random.randint(0, 255, [4, 4])
    file = os.path.join(prompts_folder, bbox_file)

    p.add_prompt(p1)
    p.add_prompt(p2)
    p.save_prompt_history(file)

    ## Force overwrite
    print(f"\nSaving {text_file} forcing overwrite.")
    p = Prompt("text")
    p1 = "HAH! YOU'VE BEEN OVERWRITTEN."
    p2 = np.random.randint(0, 255, (6, 2))
    file = os.path.join(prompts_folder, text_file)

    p.add_prompt(p1)
    p.add_prompt(p2)
    p.save_prompt_history(file, force=True)

    ## Not Force overwrite
    print(f"\nSaving {text_file} NOT forcing overwrite.")
    p = Prompt("text")
    p1 = "HAH! YOU'VE BEEN OVERWRITTEN TWICE."
    file = os.path.join(prompts_folder, text_file)

    p.add_prompt(p1)
    p.save_prompt_history(file, force=False)

    print("===== End of the Test Saving Prompts =====")

    # Test loading files
    print("===== Test Saving Prompts =====")

    ## Text prompts
    print("\nText loading.")
    p = Prompt.from_file(os.path.join(prompts_folder, text_file))
    print(p.prompts)

    ## Points prompts
    print("\nPoints loading.")
    p = Prompt.from_file(os.path.join(prompts_folder, point_file))
    print(p.prompts)

    ## Bbox prompts
    print("\nPoints loading.")
    p = Prompt.from_file(os.path.join(prompts_folder, bbox_file))
    print(p.prompts)

    print("===== End of the Test Saving Prompts =====")
