import { Component, OnInit } from '@angular/core';
import { ChatService } from '../../services/chat-service';

@Component({
  selector: 'app-chat',
  imports: [],
  templateUrl: './chat.html',
  styleUrl: './chat.css',
})
export class Chat implements OnInit {
  private list: Chat[] = [];

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    this.chatService.getChatList().subscribe((chatList) => {
      console.log(chatList);
    });
  }
}
