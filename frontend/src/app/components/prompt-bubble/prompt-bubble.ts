import { Component } from '@angular/core';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-prompt-bubble',
  imports: [FormsModule, MatFormFieldModule, MatInputModule],
  templateUrl: './prompt-bubble.html',
  styleUrl: './prompt-bubble.css',
})
export class PromptBubble {
  message: String = '';
}
