import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Chat } from './components/chat/chat';
import { PromptBubble } from './components/prompt-bubble/prompt-bubble';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Chat, PromptBubble],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected title = 'frontend';
}
